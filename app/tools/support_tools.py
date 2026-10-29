from typing import Optional
from datetime import date
from langchain_core.tools import tool

from app.core.ids import normalize_id
from app.db.database import connect
from app.services.business_rules import (
    TODAY, check_return_eligibility as business_check_return, check_return_request,
    create_return_request as write_return_request,
    create_support_ticket as write_support_ticket,
    validate_ticket, warranty_status,
)
from app.services.policy_search import search_policy as retrieve_policy


def build_tools(customer_id: str, proposal_callback, allow_proposals: bool = True):
    """Tool closures bind trusted customer identity from HTTP, never model arguments."""
    proposal_state: dict = {}
    @tool
    def get_orders(order_id: Optional[str] = None, status: Optional[str] = None, order_date: Optional[str] = None) -> dict:
        """Get only the authenticated customer's orders, including items, tracking and warranty dates. Filter by optional order_id, status or order_date (YYYY-MM-DD)."""
        with connect() as db:
            sql = "SELECT * FROM orders WHERE customer_id=?"
            params = [customer_id]
            if order_id:
                sql += " AND order_id=?"; params.append(normalize_id(order_id))
            if status:
                sql += " AND status=?"; params.append(status)
            if order_date:
                sql += " AND substr(order_date,1,10)=?"; params.append(order_date)
            orders = db.execute(sql + " ORDER BY order_date DESC", params).fetchall()
            result = []
            for order in orders:
                shipment = db.execute("SELECT * FROM shipments WHERE order_id=?", (order["order_id"],)).fetchone()
                items = db.execute("""SELECT oi.*, p.name, p.category, p.warranty_class, s.delivered_at
                    FROM order_items oi JOIN products p USING(product_id) LEFT JOIN shipments s USING(order_id)
                    WHERE oi.order_id=?""", (order["order_id"],)).fetchall()
                item_data = []
                for item in items:
                    warranty = warranty_status(db, item) if item["delivered_at"] else None
                    item_data.append({"item_id": item["item_id"], "name": item["name"], "category": item["category"], "qty": item["qty"],
                                      "opened": bool(item["opened"]),
                                      "warranty_end_date": warranty.get("warranty_end_date") if warranty else None,
                                      "warranty_active": bool(warranty and warranty.get("warranty_active")),
                                      **({"warranty_error_code": warranty["error_code"]} if warranty and warranty.get("error_code") else {})})
                delivered = shipment["delivered_at"] if shipment else None
                days = (TODAY - date.fromisoformat(delivered[:10])).days if delivered else None
                result.append({"order_id": order["order_id"], "status": order["status"], "order_date": order["order_date"],
                    "items": item_data, "shipment": dict(shipment) if shipment else None,
                    "delivery_date": delivered[:10] if delivered else None, "days_since_delivery": days})
        if order_id and not result:
            return {"error_code": "ORDER_NOT_FOUND", "message": "I couldn't find that order on your account."}
        return {"orders": result}

    @tool
    def search_policy(query: str) -> dict:
        """Search current ShopSphere policies and FAQs. Returns only applicable excerpts with citation doc_id and section. Include any date the customer mentions in the query."""
        result = retrieve_policy(query)
        if not result.get("covered") and not result.get("error"):
            result = {**result, "note": "No current ShopSphere policy covers this. Tell the customer the policies do not cover it and suggest emailing the support team. Do not guess."}
        return result

    @tool
    def check_return_eligibility(order_id: str, item_ids: list[str], reason: str) -> dict:
        """Check return eligibility deterministically. reason must be changed_mind or defective."""
        return business_check_return(customer_id, order_id, item_ids, reason)

    @tool
    def create_return_request(order_id: str, item_ids: list[str], reason: str) -> dict:
        """Propose a return request. Always ask the customer to confirm; this tool never writes before confirmation."""
        if not allow_proposals:
            return {"error_code": "ACTION_NOT_REQUESTED", "message": "The customer has not requested a new return action."}
        if proposal_state:
            return {"error_code": "ONE_PROPOSAL_AT_A_TIME", "message": "Only one write action can be proposed per message."}
        valid = check_return_request(customer_id, order_id, item_ids, reason)
        if not valid["eligible"]:
            return valid
        order_id, item_ids = normalize_id(order_id), [normalize_id(item) for item in item_ids]
        proposal = {"kind": "return", "order_id": order_id, "item_ids": item_ids, "reason": reason}
        proposal_state.update(proposal)
        proposal_callback(proposal)
        return {"pending_confirmation": True, "proposal": proposal, "message": "Eligible. Ask the customer to confirm before creating the request."}

    @tool
    def create_support_ticket(order_id: str, ticket_type: str, summary: str) -> dict:
        """Propose a warranty or cancellation support ticket. Always request explicit confirmation; never write immediately."""
        if not allow_proposals:
            return {"error_code": "ACTION_NOT_REQUESTED", "message": "The customer has not requested a new ticket action."}
        if proposal_state:
            return {"error_code": "ONE_PROPOSAL_AT_A_TIME", "message": "Only one write action can be proposed per message."}
        valid = validate_ticket(customer_id, order_id, ticket_type, summary)
        if not valid["eligible"]:
            return valid
        order_id = normalize_id(order_id)
        proposal = {"kind": "ticket", "order_id": order_id, "ticket_type": ticket_type, "summary": summary}
        proposal_state.update(proposal)
        proposal_callback(proposal)
        return {"pending_confirmation": True, "proposal": proposal, "message": "Eligible. Ask the customer to confirm before opening the ticket."}

    return [get_orders, search_policy, check_return_eligibility, create_return_request, create_support_ticket]


def execute_proposal(customer_id: str, proposal: dict):
    if proposal["kind"] == "return":
        # Revalidates all terms immediately before insertion.
        return write_return_request(customer_id, proposal["order_id"], proposal["item_ids"], proposal["reason"])
    return write_support_ticket(customer_id, proposal["order_id"], proposal["ticket_type"], proposal["summary"])
