# C1001 products and order-count audit

Audit date: 2026-10-25  
Database: configured SQLite database, queried read-only for counts. The two live HTTP conversations used a temporary copy.

## SQLite ground truth

| Measure | Database value |
|---|---:|
| Distinct orders for C1001 | 10 |
| Delivered orders | 9 |
| Cancelled orders | 1 |
| All order-item rows | 13 |
| Sum of `order_items.qty`, all statuses | 13 |
| Delivered order-item rows | 12 |
| Sum of delivered item quantities | 12 |
| Cancelled order-item rows | 1 |
| Sum of cancelled item quantities | 1 |
| Distinct product types in delivered orders | 11 |

One delivered product type, Orbitra Screen Guard X5, occurs on two separate delivered item rows. Every item row has quantity 1. The cancelled order is SS-10141 and contains one Lumio Smart Bulb 4-Pack item; another Lumio Smart Bulb 4-Pack occurs in a delivered order.

## Live HTTP replay

Both customer messages were sent to `/chat` with `X-Customer-Id: C1001` and the same session ID. The configured `gpt-6-luna` model and LangGraph agent ran against a temporary database copy.

1. **“what are products i purchased?”** — HTTP 200, route `answer`, `get_orders` recorded `success`, two real model calls. The answer listed all 11 delivered product types, marked Orbitra Screen Guard X5 as `(2)`, and separately identified the cancelled Lumio Smart Bulb order. It is complete and accurate as a distinct-product list with the duplicate quantity represented.
2. **“check again how many orders? i think 10 only.”** — HTTP 200, route `answer`, a second successful `get_orders` call with unfiltered arguments, two real model calls. The tool is bound to the authenticated C1001 customer in `build_tools`; the count did not rely solely on the suggested number. The bot answered 10 total orders, 9 delivered and 1 cancelled, and 12 delivered items. These counts match SQLite. Its “You’re right” wording was backed by the fresh lookup, so this was not unsupported agreement.
3. **Cross-customer session reuse** — HTTP 403, no model call and no tool call. The session remained customer-bound.

The first answer distinguished delivered purchases from the cancelled product. It listed 11 distinct product types, not 11 item rows: the repeated type was marked with quantity 2. The second answer distinguished 10 orders from the 12 delivered items. SQLite has 13 item rows and total quantity 13 when the cancelled row is included; the bot did not claim those were 12.

## Actual OpenAI usage

All four model calls reported usage; no token values were estimated. The provider reported 5,648 cached input tokens in total and zero reasoning tokens. Cached input is a subset of input tokens, not an additional token total.

| Query | Calls | Input tokens | Output tokens | Total tokens | Cached input tokens | Reasoning tokens | Provider call durations |
|---|---:|---:|---:|---:|---:|---:|---|
| Products question | 2 | 3,289 | 141 | 3,430 | 2,756 | 0 | 2,305.5 ms; 2,034.5 ms |
| Order-count follow-up | 2 | 3,561 | 74 | 3,635 | 2,892 | 0 | 1,489.9 ms; 2,013.5 ms |
| **Total** | **4** | **6,850** | **215** | **7,065** | **5,648** | **0** | — |

HTTP response times were 4,520.0 ms for the product list, 3,545.2 ms for the count follow-up, and 25.8 ms for the cross-customer rejection. The original database’s return-request and support-ticket counts were unchanged at zero.

## Verdict

No application defect was reproduced, so `app/agents/support_agent.py` and the database were not modified. The live verification harness is `tests/run_c1001_products_live.py`.
