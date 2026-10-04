"""执行会话：分批到达的半小时实测记录的可持久接续归集。

- **创建会话**：提交原 `/dispatch` 输入、完整计划快照与审计容差；服务端重算并
  逐字段核对快照（复用结算的 `_verify_snapshot`），确认未被篡改；
- **追加批次**：按时段下标成批追加，沿用现有遥测审计（audit_telemetry）逐段核对；
  首批首条首态 SOC 须在容差内接原初始 SOC，后续批次首条首态 SOC 须在容差内接
  已保存末态 SOC；实测动作偏离计划但记录自洽时仍接受（不做 SOC 断点判定）；
  任一记录异常、下标跳跃或超出计划范围 → 整批不入库并说明原因；
- **幂等与冲突**：同一起点的已保存整批记录原样重试 → 返回既有结果（不重复入库）；
  起点落在已确认区间且内容不同 → HTTP 409 冲突（内容不同的并发上报仅一份生效）；
- **查询**：返回原计划信息、连续已确认记录与下一待确认下标；未完成前缀可直接
  交给 `/dispatch/replan`，满期记录可直接交给 `/dispatch/settlement`；
- 会话状态持久化于 SQLite（见 session_store.py），进程重启后记录仍在。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from app.audit import audit_telemetry
from app.models import TelemetryRecord, _validate_telemetry_records
from app.session_store import SessionStore
from app.settlement import _round6, _verify_snapshot
from app.solver import PERIOD_HOURS, solve_dispatch


class SessionError(Exception):
    """执行会话请求整体拒绝：code 决定 error 字段，http_status 决定 HTTP 状态码。"""

    def __init__(self, code: str, reason: str, http_status: int = 422):
        super().__init__(reason)
        self.code = code
        self.http_status = http_status


def _status_of(next_index: int, num_periods: int) -> str:
    if next_index <= 0:
        return "open"
    if next_index < num_periods:
        return "in_progress"
    return "completed"


def _state_response(state: dict) -> dict:
    """会话状态视图：创建与查询共用。"""
    n = state["num_periods"]
    nxt = state["next_index"]
    return {
        "session_id": state["session_id"],
        "created_at": state["created_at"],
        "status": _status_of(nxt, n),
        "num_periods": n,
        "period_hours": PERIOD_HOURS,
        "audit_tolerance_kwh": state["audit_tolerance_kwh"],
        "dispatch": state["dispatch"],
        "plan_snapshot": state["plan_snapshot"],
        "confirmed_periods": nxt,
        "confirmed_records": state["records"],
        "next_index": nxt,
        "remaining_periods": n - nxt,
        "initial_soc_kwh": state["dispatch"]["initial_soc_kwh"],
        "last_soc_end_kwh": state["last_soc_end_kwh"],
        "ready_for_replan": 0 < nxt < n,
        "ready_for_settlement": nxt == n,
    }


def create_session(
    dispatch_input: dict,
    plan_snapshot: dict,
    audit_tolerance_kwh: float,
    store: SessionStore,
) -> dict:
    """创建执行会话：重算核对快照后落库，返回会话状态视图。"""
    recalculated = solve_dispatch(
        prices=dispatch_input["prices"],
        base_load_kw=dispatch_input["base_load_kw"],
        pv_kw=dispatch_input["pv_kw"],
        capacity_kwh=dispatch_input["capacity_kwh"],
        max_charge_kw=dispatch_input["max_charge_kw"],
        max_discharge_kw=dispatch_input["max_discharge_kw"],
        charge_efficiency=dispatch_input["charge_efficiency"],
        discharge_efficiency=dispatch_input["discharge_efficiency"],
        initial_soc_kwh=dispatch_input["initial_soc_kwh"],
        min_terminal_soc_kwh=dispatch_input["min_terminal_soc_kwh"],
        degradation_cost=dispatch_input["degradation_cost_per_kwh"],
    )
    _verify_snapshot(recalculated, plan_snapshot)

    state = {
        "session_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dispatch": dispatch_input,
        "plan_snapshot": plan_snapshot,
        "audit_tolerance_kwh": audit_tolerance_kwh,
        "num_periods": len(plan_snapshot["periods"]),
        "records": [],
        "batches": [],
        "next_index": 0,
        "last_soc_end_kwh": None,
    }
    store.create(state)
    return _state_response(state)


def get_session_state(session_id: str, store: SessionStore) -> dict:
    """查询会话状态；不存在返回 session_not_found（404）。"""
    state = store.get(session_id)
    if state is None:
        raise SessionError(
            "session_not_found", f"执行会话 {session_id} 不存在", http_status=404
        )
    return _state_response(state)


def append_records(
    session_id: str,
    start_index: int,
    records: Sequence[TelemetryRecord],
    store: SessionStore,
) -> dict:
    """成批追加实测记录；幂等重试返回既有结果，冲突/异常整批拒绝。"""

    def _mutate(state: dict) -> Tuple[Optional[dict], dict]:
        return _apply_batch(state, start_index, records)

    state, result = store.update(session_id, _mutate)
    if state is None:
        raise SessionError(
            "session_not_found", f"执行会话 {session_id} 不存在", http_status=404
        )
    return result


def _apply_batch(
    state: dict, start_index: int, records: Sequence[TelemetryRecord]
) -> Tuple[Optional[dict], dict]:
    """在会话状态上应用一个批次（在 store 的锁 + 事务内执行，见模块文档字符串）。"""
    n = state["num_periods"]
    next_index = state["next_index"]
    tol = state["audit_tolerance_kwh"]
    k = len(records)
    dumped = [r.model_dump() for r in records]

    # 1. 下标跳跃：起点越过当前待确认下标
    if start_index > next_index:
        raise SessionError(
            "batch_index_gap",
            f"批次起始下标 {start_index} 越过会话当前待确认下标 {next_index}，存在下标跳跃；"
            f"实测记录须按时段下标连续追加，请从下标 {next_index} 起提交，本批未入库",
        )

    # 2. 起点落在已确认区间：整批原样重试 → 返回既有结果；内容不同 → 冲突
    if start_index < next_index:
        for batch in state["batches"]:
            if batch["start_index"] == start_index:
                if batch["records"] == dumped:
                    return None, batch["response"]
                break
        raise SessionError(
            "batch_conflict",
            f"下标 {start_index} 起的记录已确认入库，本次提交与已保存批次内容不一致；"
            f"同一起点的上报仅首份生效（内容不同的并发上报返回冲突），"
            f"如需续传请从下标 {next_index} 起提交",
            http_status=409,
        )

    # 3. 超出计划范围
    if start_index + k > n:
        raise SessionError(
            "batch_out_of_range",
            f"批次覆盖下标 [{start_index}, {start_index + k - 1}]，超出原计划范围"
            f"（共 {n} 个时段，合法下标 0–{n - 1}），整批未入库",
        )

    # 4. 记录合法性（有限/非负/SOC 不超容量；容量取会话 dispatch 口径）
    dispatch = state["dispatch"]
    _validate_telemetry_records(records, dispatch["capacity_kwh"])

    # 5. 遥测审计 + 批首衔接核对（首批接原初始 SOC，后续批次接已保存末态）
    audit = audit_telemetry(
        charge_efficiency=dispatch["charge_efficiency"],
        discharge_efficiency=dispatch["discharge_efficiency"],
        energy_tolerance_kwh=tol,
        records=records,
    )
    if start_index == 0:
        prev_soc = dispatch["initial_soc_kwh"]
        prev_label = "原初始 SOC"
    else:
        prev_soc = state["last_soc_end_kwh"]
        prev_label = f"已保存末态 SOC（全局下标 {start_index - 1}）"
    first = audit["records"][0]
    gap = records[0].soc_start_kwh - prev_soc
    first["soc_continuity_gap_kwh"] = _round6(gap)
    if abs(gap) > tol:
        first["anomalous"] = True
        first["reasons"].append(
            f"批首记录与{prev_label}不衔接：首态 {records[0].soc_start_kwh}kWh "
            f"与 {prev_soc}kWh 相差 {gap:+.6f}kWh，超容差 {tol}kWh"
        )
        audit["anomalous"] = True
        if 0 not in audit["anomalous_records"]:
            audit["anomalous_records"].insert(0, 0)

    if audit["anomalous"]:
        bad0 = audit["anomalous_records"][0]
        detail = "；".join(audit["records"][bad0]["reasons"])
        raise SessionError(
            "batch_audit_failed",
            f"批次第 {bad0} 条记录（全局下标 {start_index + bad0}）未通过遥测审计，"
            f"整批 {k} 条记录均不入库：{detail}",
        )

    # 6. 整批入库：确认记录、批次快照（供幂等重试）、下一待确认下标与末态
    entries: List[dict] = []
    for entry in audit["records"]:
        global_entry = dict(entry)
        global_entry["index"] = start_index + entry["index"]
        entries.append(global_entry)

    new_next = start_index + k
    response = {
        "session_id": state["session_id"],
        "accepted": True,
        "start_index": start_index,
        "num_records": k,
        "records": entries,
        "confirmed_periods": new_next,
        "next_index": new_next,
        "last_soc_end_kwh": dumped[-1]["soc_end_kwh"],
        "num_periods": n,
        "remaining_periods": n - new_next,
        "status": _status_of(new_next, n),
    }
    new_state = dict(state)
    new_state["records"] = state["records"] + dumped
    new_state["batches"] = state["batches"] + [
        {"start_index": start_index, "records": dumped, "response": response}
    ]
    new_state["next_index"] = new_next
    new_state["last_soc_end_kwh"] = dumped[-1]["soc_end_kwh"]
    return new_state, response
