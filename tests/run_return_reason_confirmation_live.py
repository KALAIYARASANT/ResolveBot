"""HTTP verification that a bare yes is not inferred as a return reason."""
from __future__ import annotations

import json
import shutil
import socket
import sqlite3
import tempfile
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import uvicorn
from app.core.config import get_settings


def main():
    settings=get_settings(); original_db=settings.db_path
    con=sqlite3.connect(original_db)
    try:
        before={t:con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: con.close()
    records=[]
    with tempfile.TemporaryDirectory(prefix="resolvebot-return-reason-") as td:
        test_db=Path(td)/"shopsphere.db"; shutil.copy2(original_db,test_db); settings.db_path=test_db
        from app.main import app
        from app.api.router import SESSIONS
        SESSIONS.clear()
        sock=socket.socket(); sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
        server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="critical",access_log=False))
        thread=threading.Thread(target=server.run,daemon=True); thread.start()
        deadline=time.monotonic()+15
        while not server.started and thread.is_alive() and time.monotonic()<deadline: time.sleep(.05)
        if not server.started:
            server.should_exit=True; thread.join(timeout=3); settings.db_path=original_db
            print(json.dumps({"status":"BLOCKED","reason":"Local API did not start"})); return 2

        def counts():
            db=sqlite3.connect(test_db)
            try: return {t:db.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
            finally: db.close()

        def post(label,message):
            start=time.perf_counter(); status=None; payload={}; before_turn=counts()
            req=urllib.request.Request(f"http://127.0.0.1:{port}/chat",
                data=json.dumps({"session_id":"reason-confirmation-check","message":message}).encode(),
                headers={"Content-Type":"application/json","X-Customer-Id":"C1001"},method="POST")
            try:
                with urllib.request.urlopen(req,timeout=35) as response: status=response.status; payload=json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                status=exc.code
                try: payload=json.loads(exc.read().decode())
                except Exception: payload={}
            records.append({"step":label,"http_status":status,"answer":payload.get("answer"),
                "route":payload.get("route"),"pending_confirmation":payload.get("pending_confirmation"),
                "tool_calls":payload.get("tool_calls",[]),"db_before":before_turn,"db_after":counts(),
                "http_duration_ms":round((time.perf_counter()-start)*1000,1)})

        try:
            post("initial_action_without_item_or_reason","I want to return my product")
            post("identify_item_only","The Voltix 30W Car Charger")
            post("bare_yes_to_reason_question","yes")
            post("explicit_changed_mind_reason","I changed my mind")
            post("yes_to_pending_confirmation","yes")
        finally:
            server.should_exit=True; thread.join(timeout=10); settings.db_path=original_db; SESSIONS.clear()
    con=sqlite3.connect(original_db)
    try: after={t:con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("return_requests","support_tickets")}
    finally: con.close()
    print(json.dumps({"status":"COMPLETE","steps":records,"original_db_before":before,
        "original_db_after":after,"original_db_unchanged":before==after},ensure_ascii=False,separators=(",",":")))
    return 0


if __name__=="__main__": raise SystemExit(main())
