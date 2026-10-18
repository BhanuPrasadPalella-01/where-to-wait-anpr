"""SQLite store for the control room: registered vehicles, sightings, plate reads, alerts,
reconstructed routes and an append-only audit log.

Only hotlist hits and their evidence are kept long term; ordinary plate reads are pruned to a short
rolling window (privacy: no bulk record of every car's movements). Registered vehicles need a case
number and a reason, and expire automatically (review after 28 days, as the UK ANPR service does).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "control_room.db"
EXPIRY_DAYS = 28
READ_RETENTION_S = 24 * 3600  # plate reads that matched nothing are kept for one day only

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    case_no TEXT NOT NULL,
    offence TEXT NOT NULL,
    colour TEXT, vehicle_type TEXT, make_model TEXT,
    severity INTEGER NOT NULL DEFAULT 3,
    officer TEXT NOT NULL,
    notes TEXT,
    registered_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'          -- active | closed | expired
);
CREATE INDEX IF NOT EXISTS vehicles_plate ON vehicles(plate, status);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER, plate TEXT, junction INTEGER, lon REAL, lat REAL,
    t_sim REAL, wall_time REAL, score REAL, status TEXT, why TEXT, source TEXT, thumb TEXT
);
CREATE INDEX IF NOT EXISTS sightings_vehicle ON sightings(vehicle_id, t_sim);
CREATE TABLE IF NOT EXISTS reads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera TEXT, junction INTEGER, plate TEXT, conf REAL, score REAL, hit INTEGER, wall_time REAL
);
CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER, from_sighting INTEGER, to_sighting INTEGER,
    rank INTEGER, share REAL, travel_s REAL, path TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient TEXT NOT NULL,                        -- a unit name, or 'all stations'
    kind TEXT NOT NULL,                             -- assign | sighting | intercept | escape | info
    vehicle_id INTEGER, junction INTEGER, lon REAL, lat REAL,
    message TEXT NOT NULL, created_at REAL NOT NULL, acked_at REAL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wall_time REAL NOT NULL, t_sim REAL, kind TEXT, text TEXT, actor TEXT
);
"""


class Database:
    """Thin thread-safe wrapper: the FastAPI worker threads and the clock loop share one connection."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.con.executescript(SCHEMA)
            self.con.commit()

    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.con.execute(sql, args)
            self.con.commit()
            return cur

    def _rows(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.con.execute(sql, args).fetchall()]

    # ------------------------------------------------------------ vehicles (hotlist registry)
    def register_vehicle(self, plate: str, case_no: str, offence: str, officer: str, colour: str = "",
                         vehicle_type: str = "", make_model: str = "", severity: int = 3, notes: str = "",
                         now: float | None = None, expiry_days: int = EXPIRY_DAYS) -> dict:
        now = time.time() if now is None else now
        plate = "".join(c for c in plate.upper() if c.isalnum())
        if not plate or not case_no.strip() or not offence.strip() or not officer.strip():
            raise ValueError("plate, case number, offence and officer are all required")
        if self.active_vehicle(plate):
            raise ValueError(f"{plate} is already on the hotlist")
        cur = self._exec(
            "INSERT INTO vehicles (plate, case_no, offence, colour, vehicle_type, make_model, severity, officer,"
            " notes, registered_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (plate, case_no.strip(), offence.strip(), colour.strip(), vehicle_type.strip(), make_model.strip(),
             int(max(1, min(5, severity))), officer.strip(), notes.strip(), now, now + expiry_days * 86400))
        self.audit("register", f"{plate} added to hotlist (case {case_no}, {offence})", actor=officer, now=now)
        return self.vehicle(cur.lastrowid)

    def vehicle(self, vid: int) -> dict | None:
        rows = self._rows("SELECT * FROM vehicles WHERE id = ?", (vid,))
        return rows[0] if rows else None

    def active_vehicle(self, plate: str) -> dict | None:
        rows = self._rows("SELECT * FROM vehicles WHERE plate = ? AND status = 'active'", (plate,))
        return rows[0] if rows else None

    def vehicles(self, include_inactive: bool = False, now: float | None = None) -> list[dict]:
        self.expire(now)
        q = "SELECT * FROM vehicles" + ("" if include_inactive else " WHERE status = 'active'")
        return self._rows(q + " ORDER BY severity DESC, registered_at DESC")

    def close_vehicle(self, vid: int, actor: str = "officer") -> None:
        v = self.vehicle(vid)
        if v and v["status"] == "active":
            self._exec("UPDATE vehicles SET status = 'closed' WHERE id = ?", (vid,))
            self.audit("close", f"{v['plate']} removed from hotlist (case {v['case_no']})", actor=actor)

    def expire(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        rows = self._rows("SELECT id, plate FROM vehicles WHERE status = 'active' AND expires_at <= ?", (now,))
        for r in rows:
            self._exec("UPDATE vehicles SET status = 'expired' WHERE id = ?", (r["id"],))
            self.audit("expire", f"{r['plate']} expired from hotlist", actor="system", now=now)
        return len(rows)

    def hotlist(self, now: float | None = None) -> list[str]:
        return [v["plate"] for v in self.vehicles(now=now)]

    # ------------------------------------------------------------ evidence
    def add_sighting(self, rec: dict, vehicle_id: int | None) -> int:
        lon, lat = rec["lonlat"]
        cur = self._exec(
            "INSERT INTO sightings (vehicle_id, plate, junction, lon, lat, t_sim, wall_time, score, status, why,"
            " source, thumb) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (vehicle_id, rec.get("plate"), rec["camera"], lon, lat, rec["t"], time.time(), rec.get("score"),
             rec.get("status"), rec.get("why"), rec.get("source"), rec.get("thumb")))
        return cur.lastrowid

    def sightings(self, vehicle_id: int) -> list[dict]:
        return self._rows("SELECT id, plate, junction, lon, lat, t_sim, wall_time, score, status, why, source,"
                          " thumb IS NOT NULL AS has_thumb FROM sightings WHERE vehicle_id = ? ORDER BY id",
                          (vehicle_id,))

    def sighting_thumb(self, sid: int) -> str | None:
        rows = self._rows("SELECT thumb FROM sightings WHERE id = ?", (sid,))
        return rows[0]["thumb"] if rows else None

    def add_read(self, rec: dict, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._exec("INSERT INTO reads (camera, junction, plate, conf, score, hit, wall_time) VALUES (?,?,?,?,?,?,?)",
                   (rec.get("camera"), rec.get("junction"), rec.get("plate"), rec.get("conf"), rec.get("score"),
                    int(bool(rec.get("hit"))), now))
        # privacy: plate reads that matched nothing are only kept for a short rolling window
        self._exec("DELETE FROM reads WHERE hit = 0 AND wall_time < ?", (now - READ_RETENTION_S,))

    def read_counts(self) -> dict:
        r = self._rows("SELECT COUNT(*) AS n, COALESCE(SUM(hit), 0) AS hits FROM reads")[0]
        return {"reads": r["n"], "hits": r["hits"]}

    def add_routes(self, vehicle_id: int | None, from_sid: int, to_sid: int, paths: list[dict]) -> None:
        for p in paths:
            self._exec("INSERT INTO routes (vehicle_id, from_sighting, to_sighting, rank, share, travel_s, path)"
                       " VALUES (?,?,?,?,?,?,?)",
                       (vehicle_id, from_sid, to_sid, p["rank"], p["share"], p["travel_s"], json.dumps(p["lonlat"])))

    def routes(self, vehicle_id: int) -> list[dict]:
        rows = self._rows("SELECT * FROM routes WHERE vehicle_id = ? ORDER BY id", (vehicle_id,))
        for r in rows:
            r["lonlat"] = json.loads(r.pop("path"))
        return rows

    # ------------------------------------------------------------ alerts
    def add_alert(self, recipient: str, kind: str, message: str, vehicle_id: int | None = None,
                  junction: int | None = None, lonlat=None, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        lon, lat = lonlat if lonlat else (None, None)
        cur = self._exec("INSERT INTO alerts (recipient, kind, vehicle_id, junction, lon, lat, message, created_at)"
                         " VALUES (?,?,?,?,?,?,?,?)", (recipient, kind, vehicle_id, junction, lon, lat, message, now))
        return self.alert(cur.lastrowid)

    def alert(self, aid: int) -> dict | None:
        rows = self._rows("SELECT * FROM alerts WHERE id = ?", (aid,))
        return rows[0] if rows else None

    def alerts(self, recipient: str | None = None, limit: int = 50) -> list[dict]:
        if recipient is None:
            return self._rows("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        return self._rows("SELECT * FROM alerts WHERE recipient IN (?, 'all units') ORDER BY id DESC LIMIT ?",
                          (recipient, limit))

    def ack_alert(self, aid: int, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        a = self.alert(aid)
        if a and a["acked_at"] is None:
            self._exec("UPDATE alerts SET acked_at = ? WHERE id = ?", (now, aid))
            self.audit("ack", f"{a['recipient']} acknowledged alert #{aid}: {a['message']}", actor=a["recipient"],
                       now=now)
        return self.alert(aid)

    # ------------------------------------------------------------ audit log
    def audit(self, kind: str, text: str, actor: str = "system", t_sim: float | None = None,
              now: float | None = None) -> None:
        self._exec("INSERT INTO audit (wall_time, t_sim, kind, text, actor) VALUES (?,?,?,?,?)",
                   (time.time() if now is None else now, t_sim, kind, text, actor))

    def audit_log(self, limit: int = 100) -> list[dict]:
        return self._rows("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))
