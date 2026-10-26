import json
import sqlite3
import uuid
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

from app.core.config import get_settings
from app.db.database import connect, owned_order, policy_parameter

TODAY = date.fromisoformat(get_settings().app_today)


def _result(ok: bool, code: str | None = None, message: str | None = None, **extra):
    return {"eligible": ok, "error_code": code, "message": message, **extra}


def _owned_item(db, customer_id, order_id, item_id):
    return db.execute("""SELECT oi.*, p.returnable, p.warranty_class, p.name, o.status,
        s.delivered_at FROM order_items oi JOIN products p USING(product_id)
        JOIN orders o USING(order_id) LEFT JOIN shipments s USING(order_id)
        WHERE oi.item_id=? AND oi.order_id=? AND o.customer_id=?""", (item_id, order_id, customer_id)).fetchone()


def _warranty_end(delivered, months):
    return date.fromisoformat(delivered[:10]) + relativedelta(months=months)


def check_return_eligibility(customer_id: str, order_id: str, item_ids: list[str], reason: str):
    with connect() as db:
        return _check_return_eligibility(db, customer_id, order_id, item_ids, reason)


def _check_return_eligibility(db, customer_id, order_id, item_ids, reason):
    order = owned_order(db, order_id, customer_id)
    if not order:
        return _result(False, "ORDER_NOT_FOUND", "I couldn't find that order on your account.")
    if not item_ids or len(set(item_ids)) != len(item_ids):
        return _result(False, "INVALID_ITEM", "Provide valid item IDs from the order.")
    rows = [_owned_item(db, customer_id, order_id, item_id) for item_id in item_ids]
    if any(row is None for row in rows):
        return _result(False, "INVALID_ITEM", "One or more items do not belong to that order.")
    if order["status"] != "delivered" or not rows[0]["delivered_at"]:
        return _result(False, "NOT_DELIVERED", "This order must be delivered before an item can be returned.")
    for item_id, row in zip(item_ids, rows):
        if row["returnable"] == "no":
            return _result(False, "NON_RETURNABLE", "This product is not eligible for return.", item_id=item_id)
    try:
        window = int(policy_parameter(db, "return_window_days"))
    except (TypeError, ValueError):
        return _result(False, "POLICY_PARAMETER_INVALID", "The return window setting is unavailable.")
    delivered = date.fromisoformat(rows[0]["delivered_at"][:10])
    if (TODAY - delivered).days > window:
        return _result(False, "RETURN_WINDOW_EXPIRED", f"The return window ended {window} days after delivery.")
    if reason not in {"changed_mind", "defective"}:
        return _result(False, "INVALID_REASON", "Choose changed_mind or defective as the return reason.")
    for item_id, row in zip(item_ids, rows):
        if reason == "changed_mind" and row["returnable"] == "unopened_only" and row["opened"]:
            return _result(False, "OPENED_NOT_RETURNABLE", "This item can only be returned unopened for a change of mind.", item_id=item_id)
    return _result(True, message="This item is eligible for return.")


def warranty_status(db: sqlite3.Connection, row):
    if row["warranty_class"] == "none":
        return None
    months_key = "extended_warranty_months" if row["warranty_class"] == "extended" else "standard_warranty_months"
    try:
        months = int(policy_parameter(db, months_key))
    except (TypeError, ValueError):
        return {"error_code": "POLICY_PARAMETER_INVALID"}
    end = _warranty_end(row["delivered_at"], months)
    return {"warranty_end_date": end.isoformat(), "warranty_active": TODAY <= end}


def check_warranty(customer_id: str, order_id: str):
    with connect() as db:
        return _check_warranty(db, customer_id, order_id)


def _check_warranty(db, customer_id: str, order_id: str):
    order = owned_order(db, order_id, customer_id)
    if not order:
        return _result(False, "ORDER_NOT_FOUND", "I couldn't find that order on your account.")
    if order["status"] != "delivered":
        return _result(False, "NOT_DELIVERED", "This order must be delivered before a warranty claim.")
    rows = db.execute("""SELECT oi.item_id, p.warranty_class, s.delivered_at FROM order_items oi
            JOIN products p USING(product_id) LEFT JOIN shipments s USING(order_id) WHERE oi.order_id=?""", (order_id,)).fetchall()
    items = []
    for row in rows:
        if row["warranty_class"] == "none":
            continue
        if not row["delivered_at"]:
            return _result(False, "NOT_DELIVERED", "Delivery information is unavailable for this order.")
        try:
            months = int(policy_parameter(db, "extended_warranty_months" if row["warranty_class"] == "extended" else "standard_warranty_months"))
        except (TypeError, ValueError):
            return _result(False, "POLICY_PARAMETER_INVALID", "The warranty setting is unavailable.")
        end = _warranty_end(row["delivered_at"], months)
        items.append({"item_id": row["item_id"], "warranty_end_date": end.isoformat(), "warranty_active": TODAY <= end})
    if not any(i["warranty_active"] for i in items):
        code = "NO_WARRANTY" if not items else "WARRANTY_EXPIRED"
        return _result(False, code, "No item in this order is currently covered by warranty.", items=items)
    return _result(True, message="At least one item is under warranty.", items=items)


def validate_ticket(customer_id: str, order_id: str, ticket_type: str, summary: str):
    if ticket_type not in {"warranty", "cancellation"}:
        return _result(False, "INVALID_TICKET_TYPE", "Ticket type must be warranty or cancellation.")
    with connect() as db:
        return _validate_ticket(db, customer_id, order_id, ticket_type, summary)


def _validate_ticket(db, customer_id, order_id, ticket_type, summary):
    order = owned_order(db, order_id, customer_id)
    if not order:
        return _result(False, "ORDER_NOT_FOUND", "I couldn't find that order on your account.")
    if ticket_type == "warranty":
        checked = _check_warranty(db, customer_id, order_id)
        if not checked["eligible"]:
            return checked
    elif order["status"] != "processing":
        return _result(False, "ORDER_ALREADY_SHIPPED", "This order can no longer be cancelled because it has shipped.")
    return _result(True, message="Ticket request is eligible.", order_id=order_id, ticket_type=ticket_type, summary=summary)


def create_return_request(customer_id: str, order_id: str, item_ids: list[str], reason: str):
    request_id = str(uuid.uuid4())
    with connect() as db:
        valid = _check_return_eligibility(db, customer_id, order_id, item_ids, reason)
        if not valid["eligible"]:
            return valid
        db.execute("INSERT INTO return_requests VALUES (?, ?, ?, ?, ?, ?, ?)",
            (request_id, order_id, customer_id, json.dumps(item_ids), reason, "requested", datetime.now().isoformat(timespec="seconds")))
        db.commit()
    return {"created": True, "request_id": request_id, "status": "requested"}


def create_support_ticket(customer_id: str, order_id: str, ticket_type: str, summary: str):
    ticket_id = str(uuid.uuid4())
    with connect() as db:
        valid = _validate_ticket(db, customer_id, order_id, ticket_type, summary)
        if not valid["eligible"]:
            return valid
        db.execute("INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticket_id, order_id, customer_id, ticket_type, summary, "open", datetime.now().isoformat(timespec="seconds")))
        db.commit()
    return {"created": True, "ticket_id": ticket_id, "status": "open"}
