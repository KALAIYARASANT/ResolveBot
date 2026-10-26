# ResolveBot Scenario Test Report

**Run date:** 2026-10-25  
**Source of truth:** `TASK.pdf`, especially Sections 2.2 to 2.6 and 3.4.  
**Final result:** 15/15 deterministic tests passed; 18/18 live HTTP checks passed. The final live suite captured 20 actual OpenAI model calls. The original database was unchanged: `return_requests=0`, `support_tickets=0` before and after.

## Test setup

- Read the complete supplied PDF and inspected the current application, SQLite schema, policy Markdown and manifest before testing.
- Ran `.venv\Scripts\python.exe -m unittest discover -s tests -v` on Python 3.14.7. The suite creates a fresh temporary copy of the database per test; it does not need an LLM.
- Ran `tests/run_live_http.py` with the configured OpenAI key, an instrumented `ChatOpenAI` callback, a real Uvicorn HTTP server on an available local port, and a temporary database copy. The key and customer/order identifiers are not included in reports.
- Live budget: `$0.05` (`RESOLVEBOT_LIVE_TEST_BUDGET_USD`, default `$0.05`). Final measured cost was `$0.00158241`; the limit was not reached. Confirmations that bypass the model and deterministic mocked tests are not counted as LLM calls.
- No packages were installed. The supplied database still has zero return and ticket rows.

## Verified defects and fixes

1. **Effective shipping policy selection.** Before the fix, a normal shipping-fee question on `APP_TODAY=2026-08-01` selected the old `shipping_policy` paragraph, even though the current `shipping_policy_update` had taken effect. `app/services/policy_search.py` now prefers the active update for shipping fee/threshold questions after its effective date. The regression test checks July 31, August 1, and superseded status.
2. **Missing order clarification.** Before the fix, the live model called unfiltered `get_orders` for a tracking question without an order ID and answered from the returned list without clarification. `app/agents/support_agent.py` now lists the authenticated customer's matching order IDs and returns `clarify` before the model can select an arbitrary order.

Both changes were regression-tested. No other application changes were made for this test task.

## Deterministic test results

Command: `python -m unittest discover -s tests -v` (15 passed).

| Test | PDF expectation | Actual result | Status |
|---|---|---|---|
| `test_return_window_boundary_and_dynamic_parameter` | Return through the configured final day; reject one day later; read changed DB value | Passed at 30/31 days and after changing the temporary DB window to 35/36 days | PASS |
| `test_return_ownership_item_reason_and_no_invalid_write` | Same ownership isolation; reject wrong/missing item and reason; do not insert on failure | Expected error codes returned; temporary return row count stayed unchanged | PASS |
| `test_returnable_classes_and_order_status` | Enforce nonreturnable, opened-only and delivered-state rules; defective item is not rejected by the unopened-only rule | All cases matched the Section 2.3 checks | PASS |
| `test_multiple_return_items_and_partial_item_selection` | Accept valid multiple-item and partial-item selections | Both selections eligible for an order discovered from the supplied data | PASS |
| `test_warranty_end_date_dynamic_months_and_calendar_arithmetic` | Standard warranty is inclusive through its end date and uses calendar months | January 31 plus one month ended February 29, 2024; day after was expired | PASS |
| `test_extended_warranty_end_boundary_and_dynamic_duration` | Extended duration comes from SQLite and end date is inclusive | Tested before/on/after end and changed duration from one to two months | PASS |
| `test_warranty_none_and_ticket_status_rules` | No-warranty and undelivered cases reject warranty; cancellation ticket only for `processing`; bad type rejects | Expected codes returned for no warranty, not delivered, bad type and shipped; processing accepted | PASS |
| `test_write_functions_use_correct_rows_and_revalidate` | Successful writes insert one correctly populated row; failed revalidation inserts none | Temporary DB had one return and one ticket; changing the return window before a second write prevented insertion | PASS |
| `test_policy_effective_dates_superseded_and_replacement_files` | Current/effective date filters use configured files; future docs are not used early; superseded docs are excluded | July 31 excluded update; Aug 1 selected update for fee query; superseded update omitted; replacement Markdown/manifest read | PASS |
| `test_policy_topics_and_uncovered_question` | Retrieve policy/FAQ coverage without inventing unsupported policy | Returns, refunds, warranty, shipping, delay, damage, exchange, cancellation, Plus and tracking queries found expected document; unsupported query returned no excerpts/citations | PASS |
| `test_foreign_order_lookup_returns_same_not_found_result` | Foreign and unknown order lookups have same not-found result | Tool results matched exactly and contained no foreign order data | PASS |
| `test_missing_order_tracking_id_lists_matches_and_clarifies` | Missing/ambiguous order returns `clarify`; list matching orders rather than guess | Returned authenticated customer's order IDs and `clarify` | PASS |
| `test_confirmation_is_next_message_same_session_and_idempotent` | Pending proposal is session-bound, next-message-only, and not duplicated | Mocked model boundary verified session isolation, "no" cancellation, confirmation and repeated-yes idempotency | PASS |
| `test_no_unrelated_and_imperative_messages_never_execute_pending_action` | "No", unrelated message and "just do it" do not execute a pending write | No tool calls or database changes; imperative returned `clarify` | PASS |
| `test_yearless_dates_use_2026_without_rewriting_explicit_year` | Yearless dates use 2026; explicit year stays intact | Passed for month-first, day-first and explicit-year inputs | PASS |

## Live HTTP scenario results

All live requests used the API over HTTP and returned within the 30-second requirement. The live suite used only a temporary database. `R/T` below means row counts for `return_requests` / `support_tickets`; identifiers and answer wording are intentionally omitted.

| Scenario / request | Expected vs actual | HTTP / route | Model calls | Temporary DB R/T before -> after | Result |
|---|---|---:|---:|---:|---|
| API validation (`invalid_json`) | Invalid JSON gives HTTP 400 | HTTP 400 | 0 | 0/0 -> 0/0 | PASS |
| API validation (`missing_message`) | Missing required message gives HTTP 400 | HTTP 400 | 0 | 0/0 -> 0/0 | PASS |
| API validation (`missing_customer_header`) | Missing customer header gives HTTP 401 | HTTP 401 | 0 | 0/0 -> 0/0 | PASS |
| API validation (`unknown_customer_header`) | Unknown customer gives HTTP 401 | HTTP 401 | 0 | 0/0 -> 0/0 | PASS |
| order tracking (`order_tracking`) | Owned order lookup succeeds with valid response fields | HTTP 200 / `answer` | 2 | 0/0 -> 0/0 | PASS |
| session isolation (`session_customer_reuse`) | Cross-customer session reuse gives HTTP 403 | HTTP 403 | 0 | 0/0 -> 0/0 | PASS |
| foreign order isolation (`foreign_order_lookup`) | Foreign order gives the exact required not-found result, `no_answer`, and no citations | HTTP 200 / `no_answer` | 2 | 0/0 -> 0/0 | PASS |
| RAG current shipping fee (`current_shipping_fee`) | Current policy lookup uses `search_policy` and current policy citation | HTTP 200 / `answer` | 2 | 0/0 -> 0/0 | PASS |
| clarification (`missing_order_id`) | Missing order ID lists matching account orders and returns `clarify` | HTTP 200 / `clarify` | 0 | 0/0 -> 0/0 | PASS |
| return proposal (`return_proposal`) | Return is proposed; no row is written before confirmation | HTTP 200 / `answer` | 3 | 0/0 -> 0/0 | PASS |
| changed return details (`return_details_changed`) | Changed item details create a fresh proposal; no row is written | HTTP 200 / `answer` | 4 | 0/0 -> 0/0 | PASS |
| return confirmation (`return_confirmation`) | Same-session yes writes exactly one return row | HTTP 200 / `answer` | 0 | 0/0 -> 1/0 | PASS |
| duplicate confirmation (`duplicate_return_confirmation`) | Repeated yes does not write a second row | HTTP 200 / `answer` | 0 | 1/0 -> 1/0 | PASS |
| warranty ticket (`warranty_proposal`) | Eligible warranty ticket is proposed without writing | HTTP 200 / `answer` | 3 | 1/0 -> 1/0 | PASS |
| warranty confirmation (`warranty_confirmation`) | Same-session yes writes one support ticket | HTTP 200 / `answer` | 0 | 1/0 -> 1/1 | PASS |
| cancellation ticket (`cancellation_proposal`) | Processing-order cancellation ticket is proposed without writing | HTTP 200 / `answer` | 3 | 1/1 -> 1/1 | PASS |
| cancellation confirmation (`cancellation_confirmation`) | Same-session yes writes one support ticket | HTTP 200 / `answer` | 0 | 1/1 -> 1/2 | PASS |
| out-of-scope (`unrelated_query`) | Out-of-scope request has no tools, citations or writes | HTTP 200 / `out_of_scope` | 1 | 1/2 -> 1/2 | PASS |

The final HTTP suite had 13 successful customer conversation messages, four request-validation probes, and one rejected cross-customer session request. All 18 expected checks passed. The slowest successful customer response was 6.1 seconds. Confirmations and the deterministic clarification gate made no model calls.

## TASK.pdf assessment checks

| Requirement | Status | Evidence |
|---|---|---|
| Section 2.2: exact five tool names and responsibilities | PASS | Live tool calls exercised `get_orders`, `search_policy`, `check_return_eligibility`, `create_return_request`, and `create_support_ticket`; definitions are in `app/tools/support_tools.py`. |
| Section 2.3: Python return, warranty and cancellation decisions; DB parameters | PASS | Boundary, ownership, product class, calendar arithmetic, dynamic parameters, cancellation status, and write revalidation tests passed in `app/services/business_rules.py`. |
| Section 2.4: ten-document policy retrieval, manifest status/effective date, relevant excerpts/citations | PASS | Topic, unsupported, replacement-file, future-date, and superseded tests passed in `app/services/policy_search.py`; effective shipping update bug fixed. |
| Section 2.5: clarify ambiguity, propose then confirm in the next same-session message, discard non-yes, date interpretation, out-of-scope | PASS | Live return/item change/warranty/cancellation flows plus deterministic no/imperative/session/idempotency checks passed. |
| Section 2.6: request/response fields and required 400/401/403 errors | PASS | Live HTTP exercised 400, 401 and 403; successful responses had all six required fields and valid routes. Every successful response was under 30 seconds. |
| Section 3.4 check 1: service starts with the README command | UNVERIFIED | The current source started and served HTTP through Uvicorn on a free port. Port 8000 already had an existing listener (PID 28256), so the exact README port-8000 command could not be started independently in this run. |
| Section 3.4 check 2: response shape from Section 2.6 | PASS | All 13 successful live customer responses contained `request_id`, `answer`, `route`, `citations`, `tool_calls`, and `pending_confirmation`; routes were valid. |
| Section 3.4 check 3: no database writes without customer yes | PASS | Return, changed-detail, warranty and cancellation proposals left counts unchanged; writes occurred only after the next same-session affirmative. Repeated return confirmation left counts unchanged. |
| Section 3.4 check 4: return decisions under changed numbers | PASS | Temporary SQLite return window changed from 30 to 35 and boundary decisions followed the DB values. |
| Section 3.4 check 5: customer isolation | PASS | Foreign order gave same not-found response as unknown order; session reuse by another customer returned 403. |

## Scope notes and remaining verification

- The evaluator is expected to supply files with the same filenames and compatible SQLite schema. Tests used a copied database with changed `policy_parameters`, plus replacement policy/manifest paths. The complete evaluator dataset/schema was not available to run.
- The exact README command on port 8000 remains unverified because that port was occupied. The live test did verify current application startup and HTTP behavior on another local port. LangChain emitted a Python 3.14 Pydantic-v1 compatibility warning; it did not prevent the suite from completing.
- Python 3.14.7 was used. Python 3.10 itself was not installed in this environment; no package installation was performed.
- Checkout shipping-price recalculation, plus-member order creation, refunds, and the data-only `damage_report_hours` value were not tested as business actions: TASK.pdf Sections 2.2 to 2.6 do not define an order-creation/checkout fee operation or add a 48-hour return-eligibility step to Section 2.3. The shipping effective-date change was tested as policy retrieval.

## Final assessment

**Tested assessment behaviors: PASS.** The one item not independently verified is the exact port-8000 README startup invocation because an existing process held that port. The two reproduced application defects were fixed and their regression tests pass. The original database remained untouched.
