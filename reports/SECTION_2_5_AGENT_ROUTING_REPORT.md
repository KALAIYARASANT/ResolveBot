# Section 2.5 agent-routing review

Run date: 2026-10-25  
Configured model: `gpt-6-luna`  
Live-test budget limit: USD 0.05  
Live HTTP tests ran against a temporary copy of the SQLite database.

## Results

| Check | Result | Live evidence |
|---|---|---|
| Normal order requests reach the OpenAI/LangGraph agent | **PASS** | ID-specific order-status request: 2 model calls; `get_orders` executed; HTTP 200, route `answer`. |
| Return eligibility requests reach the agent | **PASS** | ID-specific changed-mind eligibility request: 2 model calls; `check_return_eligibility` executed; HTTP 200, route `answer`. |
| Warranty requests reach the agent | **PASS** | Order-specific warranty request: 2 model calls; `get_orders` executed; HTTP 200, route `answer`. |
| Cancellation requests reach the agent and require confirmation | **PASS** | Cancellation proposal: 3 model calls; `get_orders` and `create_support_ticket` executed; proposal pending; no write before confirmation. |
| Policy requests reach the agent | **PASS** | Return-window policy request: 2 model calls; `search_policy` executed; HTTP 200, route `answer`. |
| Deterministic broad-return clarification preserves required tool record | **PASS** | Broad return question: 0 model calls; HTTP 200, route `clarify`; `get_orders` recorded as successful; no eligibility call or proposal. |
| Over-budget model tool calls are all accounted for without being reissued | **PASS after fix** | Real multi-intent request generated 11 calls. Six ran; five over-budget eligibility calls were recorded as skipped/error. One model request only; HTTP 200, route `clarify`; no `OpenAIInvalidRequestError`. Generated and recorded call counts both 11. |
| Cross-customer confirmation protection | **PASS** | Reusing the pending cancellation session as another customer returned HTTP 403, with no model call or write. |
| Write-time revalidation | **PASS** | After the proposal, the temporary order was changed from `processing` to `shipped`. The owner’s next-message confirmation was rejected by the Python cancellation rule; no support ticket was inserted. |

## Confirmed boundary defect and minimal fix

The first live multi-intent run exposed a gap at the cumulative tool-call limit. After six calls had executed, the model could emit a further batch. The graph ended when it saw that the execution budget was exhausted, leaving those newly emitted calls without tool records. They were not sent back to OpenAI, so that run did not produce HTTP 400, but the response could be incomplete.

In `app/agents/support_agent.py:260`, the graph now routes every assistant message containing tool calls through `tools_node`, even when the remaining execution budget is zero. The node records a safe limit result for each unexecuted call and emits a matching `ToolMessage`; `agent_node` then returns the deterministic limit clarification without another model request. The cumulative-boundary regression is covered by `tests/test_scenarios.py:448`.

## Live OpenAI usage and response timing

Provider usage was captured for every actual model call. No usage was missing; no mocked calls are included. Cached-input and reasoning tokens were provider-reported as zero for these calls.

| Customer query | Calls | Input | Output | Total | HTTP duration |
|---|---:|---:|---:|---:|---:|
| Broad return clarification | 0 | 0 | 0 | 0 | 41.2 ms |
| Order status | 2 | 1,330 | 77 | 1,407 | 4,224.3 ms |
| Return eligibility | 2 | 1,177 | 69 | 1,246 | 4,107.9 ms |
| Warranty | 2 | 1,325 | 53 | 1,378 | 4,569.4 ms |
| Policy | 2 | 1,389 | 79 | 1,468 | 5,115.8 ms |
| Complex multi-intent, 11 generated tool calls | 1 | 630 | 356 | 986 | 3,717.7 ms |
| Cancellation proposal | 3 | 2,123 | 113 | 2,236 | 7,068.4 ms |
| Cross-customer confirmation | 0 | 0 | 0 | 0 | 22.2 ms |
| Owner confirmation after invalidating order status | 0 | 0 | 0 | 0 | 27.1 ms |
| **Total** | **12** | **7,974** | **747** | **8,721** | — |

Provider call durations, in order by query:

- Order status: 2,202.1 ms; 1,864.2 ms.
- Return eligibility: 2,445.5 ms; 1,582.9 ms.
- Warranty: 2,298.6 ms; 2,233.7 ms.
- Policy: 2,634.1 ms; 2,426.3 ms.
- Complex multi-intent: 3,623.0 ms.
- Cancellation proposal: 2,736.7 ms; 2,487.2 ms; 1,788.6 ms.

The longest complete HTTP response took 7,068.4 ms, within the 30-second requirement. The live runner reported all expected HTTP statuses, no OpenAI invalid-request errors, and complete tool-call records. The original database still contained zero return requests and zero support tickets after the run.

## Files changed for this review

- `app/agents/support_agent.py`: routes exhausted-budget tool batches through safe result recording.
- `tests/test_scenarios.py`: adds a cumulative-limit regression test.
- `tests/run_agent_routing_live.py`: targeted live HTTP/OpenAI routing, budget, token, and revalidation capture.
