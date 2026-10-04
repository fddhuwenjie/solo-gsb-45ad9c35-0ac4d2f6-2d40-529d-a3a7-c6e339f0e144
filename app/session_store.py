"""执行会话的持久化存储（SQLite 单文件）。

设计要点：
- 整份会话状态（原计划输入、计划快照、审计容差、连续已确认记录、批次快照、
  下一待确认下标、已保存末态 SOC）以 JSON 单记录保存；进程重启后重新打开同一
  数据库文件即可恢复全部记录；
- 「读-改-写」在进程内锁 + BEGIN IMMEDIATE 事务内完成：进程内多线程串行，
  跨进程由 SQLite 忙等待（connect timeout）兜底，保证同一起点内容不同的并发
  上报只有一份生效。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import closing
from typing import Callable, Optional, Tuple

# 默认数据库位置（相对工作目录）；可用环境变量 SESSION_DB_PATH 覆盖。
DEFAULT_DB_PATH = os.path.join("data", "sessions.db")


class SessionStore:
    """会话状态存储：每行一个会话，state_json 为完整会话状态。"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # timeout 为 SQLite 忙等待秒数：跨进程写冲突时等待而非立即报错
        return sqlite3.connect(self._db_path, timeout=10.0)

    def create(self, state: dict) -> None:
        """写入新会话（session_id 为随机 uuid，不会冲突）。"""
        payload = json.dumps(state, ensure_ascii=False)
        with self._lock, closing(self._connect()) as conn:
            with conn:  # 正常提交、异常回滚
                conn.execute(
                    "INSERT INTO sessions (session_id, created_at, state_json) VALUES (?, ?, ?)",
                    (state["session_id"], state["created_at"], payload),
                )

    def get(self, session_id: str) -> Optional[dict]:
        """读取会话状态；不存在返回 None。"""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def update(
        self,
        session_id: str,
        fn: "Callable[[dict], Tuple[Optional[dict], object]]",
    ) -> "Tuple[Optional[dict], object]":
        """在锁 + IMMEDIATE 事务内执行「读 → fn(state) → 写回」。

        fn 返回 (new_state, result)：new_state 为 None 时不写库（如幂等重试）。
        会话不存在时返回 (None, None)；fn 抛出的异常原样传播（事务回滚，不入库）。
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return None, None
                state = json.loads(row[0])
                new_state, result = fn(state)
                if new_state is not None:
                    conn.execute(
                        "UPDATE sessions SET state_json = ? WHERE session_id = ?",
                        (json.dumps(new_state, ensure_ascii=False), session_id),
                    )
                conn.commit()
                return state, result
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
