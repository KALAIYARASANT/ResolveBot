"""Simple interactive client for ResolveBot's existing POST /chat endpoint."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402


CHAT_URL = "http://127.0.0.1:8000/chat"
HTTP_TIMEOUT_SECONDS = 35


def configured_db_path() -> Path:
    """Resolve relative DB_PATH values from the project root."""
    path = Path(get_settings().db_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def customer_exists_read_only(customer_id: str) -> bool:
    """Check the configured customers table without opening the DB for writes."""
    db_path = configured_db_path()
    if not db_path.is_file():
        raise FileNotFoundError(f"Configured database was not found: {db_path}")

    read_only_uri = f"{db_path.as_uri()}?mode=ro"
    db = sqlite3.connect(read_only_uri, uri=True, timeout=5)
    try:
        db.execute("PRAGMA query_only = ON")
        return db.execute(
            "SELECT 1 FROM customers WHERE customer_id = ? LIMIT 1",
            (customer_id,),
        ).fetchone() is not None
    finally:
        db.close()


def post_message(customer_id: str, session_id: str, message: str) -> tuple[str, int, dict | None]:
    """Return (answer, elapsed_ns, full API payload); the payload is None when the request failed."""
    payload = None
    body = json.dumps({"session_id": session_id, "message": message}).encode("utf-8")
    request = Request(
        CHAT_URL,
        data=body,
        headers={
            "X-Customer-Id": customer_id,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started_ns = time.perf_counter_ns()
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
            raise ValueError("The API response did not contain a readable answer.")
        answer = payload["answer"]
    except HTTPError as exc:
        detail = ""
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            detail = error_payload.get("detail", "") if isinstance(error_payload, dict) else ""
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        message_text = f"API error (HTTP {exc.code})"
        if detail:
            message_text += f": {detail}"
        answer = message_text
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        answer = f"Connection error: could not reach {CHAT_URL} ({reason})."
    except (TimeoutError, OSError) as exc:
        answer = f"Connection error: {exc}"
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        answer = f"API response error: {exc}"
    except Exception as exc:
        answer = f"Request failed: {type(exc).__name__}: {exc}"
    elapsed_ns = time.perf_counter_ns() - started_ns
    return answer, elapsed_ns, payload


def post_message_with_loading(customer_id: str, session_id: str, message: str) -> tuple[str, int, dict | None]:
    """Animate dots while the synchronous HTTP request waits for the bot."""
    stop = threading.Event()

    def show_dots() -> None:
        dots = 0
        while not stop.wait(0.35):
            dots = dots % 3 + 1
            sys.stdout.write("\rBot is typing" + "." * dots + "   ")
            sys.stdout.flush()

    indicator = threading.Thread(target=show_dots, daemon=True)
    indicator.start()
    try:
        return post_message(customer_id, session_id, message)
    finally:
        stop.set()
        indicator.join()
        sys.stdout.write("\r" + " " * 24 + "\r")
        sys.stdout.flush()


def print_json_details(payload: dict | None) -> None:
    """Show the complete API response plus its key fields, keeping Unicode such as § readable."""
    if not isinstance(payload, dict):
        print("(no JSON response body to show)")
        return
    print("--- API response (JSON) ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("--- Summary ---")
    print(f"route: {payload.get('route')}")
    print("pending_confirmation: " + json.dumps(payload.get("pending_confirmation")))
    print("citations: " + json.dumps(payload.get("citations"), ensure_ascii=False))
    print("tool_calls: " + json.dumps(payload.get("tool_calls"), ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive client for ResolveBot's POST /chat endpoint.")
    parser.add_argument("--show-json", action="store_true",
                        help="after each answer, print the complete JSON response, route, citations, tool calls and pending status")
    show_json = parser.parse_args().show_json
    if show_json and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        while True:
            customer_id = input("Enter Customer ID: ").strip()
            if customer_id.casefold() == "exit":
                return 0
            if not customer_id:
                continue
            try:
                valid = customer_exists_read_only(customer_id)
            except (OSError, sqlite3.Error) as exc:
                print(f"Could not validate customer ID: {exc}")
                return 1
            if not valid:
                print("No customer found with this ID.")
                continue
            break

        print("Customer verified successfully.")
        session_id = str(uuid.uuid4())
        print("Chat started. Type 'exit' to end the session.")
        message_number = 0

        while True:
            try:
                message = input("You: ").strip()
            except EOFError:
                print("\nSession ended.")
                return 0
            if message.casefold() == "exit":
                print("Session ended.")
                return 0
            if not message:
                continue

            message_number += 1
            if show_json:
                request_body = {"session_id": session_id, "message": message}
                print(f"Message {message_number}: {json.dumps(request_body, ensure_ascii=False)}")
            answer, elapsed_ns, payload = post_message_with_loading(customer_id, session_id, message)
            answer = answer.replace("**", "")
            print(f"Bot: {answer}")
            if show_json:
                print_json_details(payload)
            print(f"Time taken: {elapsed_ns} ns ({elapsed_ns / 1_000_000_000:.9f} seconds)")
            print()
    except KeyboardInterrupt:
        print("\nSession ended.")
        return 0
    except EOFError:
        print("\nSession ended.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
