"""执行会话接口测试：pytest tests/ -q"""

import asyncio
import copy

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.session_store import SessionStore

client = TestClient(app)


# 4 个半小时时段、效率 0.9、衰减费 0.2 元/kWh 的调度输入（与结算/续排测试同口径）。
DISPATCH = dict(
    prices=[0.0, 1.0, 1.0, 0.0],
    base_load_kw=[0.2, 0.2, 0.2, 0.2],
    pv_kw=[0.0, 0.0, 0.0, 0.0],
    capacity_kwh=0.4,
    max_charge_kw=1.0,
    max_discharge_kw=1.0,
    charge_efficiency=0.9,
    discharge_efficiency=0.9,
    initial_soc_kwh=0.0,
    min_terminal_soc_kwh=0.0,
    degradation_cost_per_kwh=0.2,
)

# 与原计划自洽的完整 4 段实测记录（计划：充0.2、放0.1、放0.1、静置）。
RECORDS = [
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=round(0.2 / 0.9, 6), discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=round(0.1 + 0.2 / 0.9, 6),
        soc_start_kwh=0.0, soc_end_kwh=0.2,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.2, soc_end_kwh=0.1,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.1, soc_end_kwh=0.0,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=0.1,
        soc_start_kwh=0.0, soc_end_kwh=0.0,
    ),
]

TOL = 0.01


@pytest.fixture()
def store(tmp_path):
    """每个用例使用独立的临时 SQLite 库，用例间互不影响。"""
    s = SessionStore(str(tmp_path / "sessions.db"))
    previous = getattr(app.state, "session_store", None)
    app.state.session_store = s
    try:
        yield s
    finally:
        if previous is None:
            try:
                del app.state.session_store
            except AttributeError:
                pass
        else:
            app.state.session_store = previous


def _plan(dispatch=None):
    dispatch = dispatch or DISPATCH
    return client.post("/dispatch", json=dispatch).json()


def _create(dispatch=None, plan=None, tol=TOL):
    dispatch = dispatch or DISPATCH
    return client.post(
        "/dispatch/sessions",
        json={
            "dispatch": dispatch,
            "plan_snapshot": plan or _plan(dispatch),
            "audit_tolerance_kwh": tol,
        },
    )


def _new_session(store):
    r = _create()
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def _append(sid, start_index, records):
    return client.post(
        f"/dispatch/sessions/{sid}/records",
        json={"start_index": start_index, "records": records},
    )


def _get(sid):
    return client.get(f"/dispatch/sessions/{sid}")


# ---------- 创建会话 ----------


def test_create_session_returns_initial_state(store):
    r = _create()
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["session_id"]
    assert body["status"] == "open"
    assert body["num_periods"] == 4
    assert body["period_hours"] == 0.5
    assert body["audit_tolerance_kwh"] == TOL
    assert body["confirmed_periods"] == 0
    assert body["confirmed_records"] == []
    assert body["next_index"] == 0
    assert body["remaining_periods"] == 4
    assert body["initial_soc_kwh"] == 0.0
    assert body["last_soc_end_kwh"] is None
    assert body["ready_for_replan"] is False
    assert body["ready_for_settlement"] is False
    # 原计划信息原样回显
    assert body["dispatch"]["prices"] == DISPATCH["prices"]
    assert body["plan_snapshot"]["summary"]["num_periods"] == 4


def test_create_session_tampered_snapshot_rejected(store):
    tampered = copy.deepcopy(_plan())
    tampered["periods"][0]["energy_cost_yuan"] = 9.9
    r = _create(plan=tampered)
    assert r.status_code == 422
    assert r.json()["error"] == "plan_snapshot_mismatch"


def test_create_session_infeasible_dispatch_rejected(store):
    bad = dict(DISPATCH, max_charge_kw=0.0, min_terminal_soc_kwh=0.4)
    r = _create(dispatch=bad, plan=_plan())
    assert r.status_code == 422
    assert r.json()["error"] == "infeasible"


def test_create_session_invalid_tolerance_rejected(store):
    assert _create(tol=-0.01).status_code == 422
    r = client.post(
        "/dispatch/sessions",
        content='{"dispatch": %s, "plan_snapshot": %s, "audit_tolerance_kwh": NaN}'
        % (
            __import__("json").dumps(DISPATCH),
            __import__("json").dumps(_plan()),
        ),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"


# ---------- 追加与查询 ----------


def test_append_batches_in_order_then_query(store):
    sid = _new_session(store)

    r1 = _append(sid, 0, copy.deepcopy(RECORDS[:2]))
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["accepted"] is True
    assert b1["start_index"] == 0
    assert b1["num_records"] == 2
    assert b1["next_index"] == 2
    assert b1["confirmed_periods"] == 2
    assert b1["last_soc_end_kwh"] == 0.1
    assert b1["status"] == "in_progress"
    assert b1["remaining_periods"] == 2
    # 审计结论使用全局下标；批首衔接差为 0（接原初始 SOC）
    assert [e["index"] for e in b1["records"]] == [0, 1]
    assert all(e["anomalous"] is False for e in b1["records"])
    assert b1["records"][0]["soc_continuity_gap_kwh"] == 0.0

    r2 = _append(sid, 2, copy.deepcopy(RECORDS[2:]))
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["next_index"] == 4
    assert b2["status"] == "completed"
    assert b2["last_soc_end_kwh"] == 0.0
    assert [e["index"] for e in b2["records"]] == [2, 3]

    q = _get(sid)
    assert q.status_code == 200
    body = q.json()
    assert body["status"] == "completed"
    assert body["confirmed_periods"] == 4
    assert body["confirmed_records"] == RECORDS
    assert body["next_index"] == 4
    assert body["remaining_periods"] == 0
    assert body["last_soc_end_kwh"] == 0.0
    assert body["ready_for_settlement"] is True
    assert body["ready_for_replan"] is False


def test_first_batch_first_state_must_match_initial_soc(store):
    """首批首态未在容差内接原初始 SOC → batch_audit_failed，整批不入库。"""
    sid = _new_session(store)
    records = copy.deepcopy(RECORDS[:2])
    records[0]["soc_start_kwh"] = 0.1  # 与原初始 SOC 0.0 相差 0.1 > 0.01
    r = _append(sid, 0, records)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "batch_audit_failed"
    assert "原初始 SOC" in body["reason"]
    # 整批未入库
    assert _get(sid).json()["next_index"] == 0
    assert _get(sid).json()["confirmed_records"] == []

    # 边界：偏离恰好等于容差 → 接受（严格大于才判超差）
    idle = dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=0.1,
        soc_start_kwh=0.01, soc_end_kwh=0.01,
    )
    r = _append(sid, 0, [idle])
    assert r.status_code == 200, r.text
    assert r.json()["records"][0]["soc_continuity_gap_kwh"] == 0.01


def test_later_batch_first_state_must_match_saved_last(store):
    """后续批次首态未接已保存末态 → batch_audit_failed，整批不入库。"""
    sid = _new_session(store)
    assert _append(sid, 0, copy.deepcopy(RECORDS[:1])).status_code == 200  # 末态 0.2

    bad = dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=0.1,
        soc_start_kwh=0.15, soc_end_kwh=0.15,  # 与已保存末态 0.2 相差 0.05 > 0.01
    )
    r = _append(sid, 1, [bad])
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "batch_audit_failed"
    assert "已保存末态 SOC" in body["reason"]
    # 整批未入库，待确认下标不变
    state = _get(sid).json()
    assert state["next_index"] == 1
    assert state["confirmed_periods"] == 1
    assert state["last_soc_end_kwh"] == 0.2


def test_deviation_from_plan_self_consistent_accepted(store):
    """实测动作偏离计划但自洽：照常接受（不做 SOC 断点判定）。"""
    sid = _new_session(store)
    # t0 原计划充 0.2；实际只充 0.1（能量/购电自洽）
    deviated = dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=round(0.1 / 0.9, 6), discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=round(0.1 + 0.1 / 0.9, 6),
        soc_start_kwh=0.0, soc_end_kwh=0.1,
    )
    r = _append(sid, 0, [deviated])
    assert r.status_code == 200, r.text
    assert r.json()["last_soc_end_kwh"] == 0.1
    state = _get(sid).json()
    assert state["confirmed_periods"] == 1
    assert state["confirmed_records"][0]["soc_end_kwh"] == 0.1


def test_index_gap_rejected(store):
    """下标跳跃 → batch_index_gap，整批不入库。"""
    sid = _new_session(store)
    r = _append(sid, 1, copy.deepcopy(RECORDS[1:2]))
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "batch_index_gap"
    assert "下标 0" in body["reason"]
    assert _get(sid).json()["next_index"] == 0


def test_out_of_range_rejected(store):
    """批次覆盖下标超出原计划范围 → batch_out_of_range，整批不入库。"""
    sid = _new_session(store)
    assert _append(sid, 0, copy.deepcopy(RECORDS[:2])).status_code == 200
    r = _append(sid, 2, copy.deepcopy(RECORDS[2:]) + [copy.deepcopy(RECORDS[3])])
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "batch_out_of_range"
    assert _get(sid).json()["next_index"] == 2

    # 满期后继续追加同样按超范围拒绝
    assert _append(sid, 2, copy.deepcopy(RECORDS[2:])).status_code == 200
    r = _append(sid, 4, [copy.deepcopy(RECORDS[3])])
    assert r.status_code == 422
    assert r.json()["error"] == "batch_out_of_range"


def test_anomalous_record_rejects_whole_batch(store):
    """批内任一记录异常 → 整批（含正常记录）不入库。"""
    sid = _new_session(store)
    records = copy.deepcopy(RECORDS[:2])
    records[1]["grid_purchase_energy_kwh"] = 0.11  # 期望 0.01，残差 0.1 超容差
    r = _append(sid, 0, records)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "batch_audit_failed"
    assert "全局下标 1" in body["reason"]
    assert "购电量残差" in body["reason"]
    # 第一条记录本身正常，但整批不入库
    state = _get(sid).json()
    assert state["next_index"] == 0
    assert state["confirmed_records"] == []


def test_invalid_record_values_rejected(store):
    """测量值为负 / SOC 超容量沿用既有校验 → invalid_input。"""
    sid = _new_session(store)
    records = copy.deepcopy(RECORDS[:1])
    records[0]["grid_purchase_energy_kwh"] = -0.1
    assert _append(sid, 0, records).status_code == 422

    records = copy.deepcopy(RECORDS[:1])
    records[0]["soc_end_kwh"] = 0.5  # 超容量 0.4
    r = _append(sid, 0, records)
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"

    # 空批次与负下标由请求模型拦截
    assert _append(sid, 0, []).status_code == 422
    assert _append(sid, -1, copy.deepcopy(RECORDS[:1])).status_code == 422


# ---------- 幂等重试与冲突 ----------


def test_idempotent_retry_returns_existing_result(store):
    """同一起点的已保存整批记录原样重试 → 返回既有结果，不重复入库。"""
    sid = _new_session(store)
    r1 = _append(sid, 0, copy.deepcopy(RECORDS[:2]))
    assert r1.status_code == 200
    r2 = _append(sid, 0, copy.deepcopy(RECORDS[:2]))
    assert r2.status_code == 200
    assert r2.json() == r1.json()  # 既有结果原样返回
    assert _get(sid).json()["confirmed_periods"] == 2  # 未重复入库

    # 满期批次同样可幂等重试
    r3 = _append(sid, 2, copy.deepcopy(RECORDS[2:]))
    assert r3.status_code == 200
    r4 = _append(sid, 2, copy.deepcopy(RECORDS[2:]))
    assert r4.status_code == 200
    assert r4.json() == r3.json()
    assert _get(sid).json()["confirmed_periods"] == 4


def test_conflicting_content_rejected(store):
    """起点落在已确认区间且内容不同 → 409 batch_conflict。"""
    sid = _new_session(store)
    assert _append(sid, 0, copy.deepcopy(RECORDS[:2])).status_code == 200

    # 同起点、内容不同
    tampered = copy.deepcopy(RECORDS[:2])
    tampered[0]["grid_purchase_energy_kwh"] = 0.4
    r = _append(sid, 0, tampered)
    assert r.status_code == 409
    assert r.json()["error"] == "batch_conflict"

    # 起点落在已保存批次中间（无整批对应）同样冲突
    r = _append(sid, 1, copy.deepcopy(RECORDS[1:2]))
    assert r.status_code == 409
    assert r.json()["error"] == "batch_conflict"

    # 已确认内容不受影响
    assert _get(sid).json()["confirmed_periods"] == 2


def test_concurrent_conflicting_batches_single_winner(store):
    """内容不同的并发上报：只有一份生效，其余返回冲突。"""
    sid = _new_session(store)
    batch_a = [copy.deepcopy(RECORDS[0])]
    # 另一份同样自洽但内容不同（实际只充 0.1）
    batch_b = [
        dict(
            base_load_kw=0.2, pv_kw=0.0,
            charge_ac_energy_kwh=round(0.1 / 0.9, 6), discharge_ac_energy_kwh=0.0,
            grid_purchase_energy_kwh=round(0.1 + 0.1 / 0.9, 6),
            soc_start_kwh=0.0, soc_end_kwh=0.1,
        )
    ]

    async def _post_both():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as ac:
            return await asyncio.gather(
                ac.post(
                    f"/dispatch/sessions/{sid}/records",
                    json={"start_index": 0, "records": batch_a},
                ),
                ac.post(
                    f"/dispatch/sessions/{sid}/records",
                    json={"start_index": 0, "records": batch_b},
                ),
            )

    ra, rb = asyncio.run(_post_both())
    assert sorted([ra.status_code, rb.status_code]) == [200, 409]
    conflict = ra if ra.status_code == 409 else rb
    assert conflict.json()["error"] == "batch_conflict"
    # 只有一份生效
    state = _get(sid).json()
    assert state["confirmed_periods"] == 1
    winner = ra if ra.status_code == 200 else rb
    assert state["confirmed_records"][0]["soc_end_kwh"] == (
        0.2 if winner.json()["last_soc_end_kwh"] == 0.2 else 0.1
    )


# ---------- 会话不存在 ----------


def test_session_not_found(store):
    r = _get("no-such-session")
    assert r.status_code == 404
    assert r.json()["error"] == "session_not_found"

    r = _append("no-such-session", 0, copy.deepcopy(RECORDS[:1]))
    assert r.status_code == 404
    assert r.json()["error"] == "session_not_found"


# ---------- 与续排 / 结算衔接 ----------


def test_confirmed_prefix_feeds_replan(store):
    """未完成前缀可直接作为 /dispatch/replan 的 executed_records。"""
    sid = _new_session(store)
    assert _append(sid, 0, copy.deepcopy(RECORDS[:2])).status_code == 200
    state = _get(sid).json()
    assert state["ready_for_replan"] is True
    assert state["ready_for_settlement"] is False

    r = client.post(
        "/dispatch/replan",
        json={
            "dispatch": state["dispatch"],
            "plan_snapshot": state["plan_snapshot"],
            "audit_tolerance_kwh": state["audit_tolerance_kwh"],
            "executed_records": state["confirmed_records"],
            "remaining_prices": [5.0, 0.0],
            "remaining_base_load_kw": [0.2, 0.2],
            "remaining_pv_kw": [0.0, 0.0],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["executed_periods"] == 2
    assert r.json()["executed_summary"]["final_soc_kwh"] == 0.1


def test_completed_session_feeds_settlement(store):
    """满期记录可直接作为 /dispatch/settlement 的 records。"""
    sid = _new_session(store)
    assert _append(sid, 0, copy.deepcopy(RECORDS[:2])).status_code == 200
    assert _append(sid, 2, copy.deepcopy(RECORDS[2:])).status_code == 200
    state = _get(sid).json()
    assert state["ready_for_settlement"] is True

    r = client.post(
        "/dispatch/settlement",
        json={
            "dispatch": state["dispatch"],
            "plan_snapshot": state["plan_snapshot"],
            "settlement_tolerance_kwh": state["audit_tolerance_kwh"],
            "records": state["confirmed_records"],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "settled"
    assert r.json()["settled_periods"] == 4


# ---------- 进程重启后记录仍在 ----------


def test_records_survive_process_restart(store, tmp_path):
    """重新打开同一 SQLite 文件（等价于进程重启）后，记录与进度仍在。"""
    sid = _new_session(store)
    assert _append(sid, 0, copy.deepcopy(RECORDS[:2])).status_code == 200

    # 模拟进程重启：抛弃内存中的 store 实例，重新打开同一数据库文件
    app.state.session_store = SessionStore(str(tmp_path / "sessions.db"))

    state = _get(sid).json()
    assert state["confirmed_periods"] == 2
    assert state["confirmed_records"] == RECORDS[:2]
    assert state["next_index"] == 2
    assert state["last_soc_end_kwh"] == 0.1

    # 重启后仍可继续追加、幂等重试仍然有效
    r = _append(sid, 0, copy.deepcopy(RECORDS[:2]))
    assert r.status_code == 200  # 重试返回既有结果
    r = _append(sid, 2, copy.deepcopy(RECORDS[2:]))
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


# ---------- 既有接口不受影响 ----------


def test_existing_endpoints_unchanged(store):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.post("/dispatch", json=DISPATCH).status_code == 200
    r = client.post(
        "/telemetry/audit",
        json={
            "capacity_kwh": DISPATCH["capacity_kwh"],
            "charge_efficiency": 0.9,
            "discharge_efficiency": 0.9,
            "energy_tolerance_kwh": TOL,
            "records": RECORDS,
        },
    )
    assert r.status_code == 200
    assert r.json()["anomalous"] is False
