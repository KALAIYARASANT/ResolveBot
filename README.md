# ResolveBot

ResolveBot is a small ShopSphere support API. It answers order and policy questions and can prepare a return or support ticket for creation after the customer confirms it.

## Setup and run

Use Python 3.14 (the assessment requires Python 3.10 or later). Create a virtual environment, install the pinned requirements, copy `.env.example` to `.env`, add an OpenAI API key, and run the API:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

PowerShell request:

```powershell
$body = @{ session_id = 's1'; message = 'Where is order SS-10512?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -Headers @{'X-Customer-Id'='C1001'} -ContentType 'application/json' -Body $body
```

curl request (Windows):

```powershell
curl.exe -X POST http://localhost:8000/chat `
  -H "X-Customer-Id: C1001" -H "Content-Type: application/json" `
  -d '{"session_id":"s1","message":"Where is order SS-10512?"}'
```

The only endpoint is `POST /chat` on port 8000. It requires `X-Customer-Id` and a JSON body with `session_id` and `message`. Each successful response has a unique `request_id`, `answer`, one of `answer`, `clarify`, `no_answer`, or `out_of_scope`, citations, chronologically ordered tool calls, and `pending_confirmation`.

Each `tool_calls` entry is `{"name": ..., "args": {...}, "status": "success" | "error" | "pending_confirmation"}`, listing the tools actually run for that message, in order. `args` is exactly what the tool received (for example `get_orders` called without a filter has `"args": {}`). A confirmed write returns `create_return_request` or `create_support_ticket` with status `success` and `pending_confirmation: false`. JSON keeps `§` as a readable character.

To inspect the full response while chatting in the terminal, start the script with `--show-json`; it prints the complete JSON, route, citations, tool calls, and pending status after each answer:

```powershell
.venv\Scripts\python.exe scripts\chat_terminal.py --show-json
```

## Architecture

- `app/main.py` creates the FastAPI application.
- `app/api/router.py` validates HTTP input, customer identity, and session ownership.
- `app/schemas/chat.py` defines request and response data.
- `app/core/config.py` loads environment configuration.
- `app/agents/support_agent.py` runs one LangGraph model/tool loop, with a six-call maximum and session handling.
- `app/tools/support_tools.py` defines the five tools and binds customer identity from trusted Python context.
- `app/services/business_rules.py` implements return and warranty decisions and revalidates writes.
- `app/services/policy_search.py` reads policy documents and manifest entries, then retrieves relevant current paragraphs.
- `app/db/database.py` contains small SQLite connection and lookup helpers.

The tools are `get_orders(order_id=None, status=None)`, `search_policy(query)`, `check_return_eligibility(order_id, item_ids, reason)`, `create_return_request(order_id, item_ids, reason)`, and `create_support_ticket(order_id, ticket_type, summary)`. The model chooses tools; Python checks all authority-sensitive and eligibility-sensitive data. The customer ID is never a model argument.

## Policy retrieval and citations

The retriever splits the configured Markdown file by document ID and numbered headings. It matches each document against `manifest.json`, excludes superseded policies, and excludes future-effective notices from current-policy questions using `APP_TODAY`. For an explicitly future-dated question, it can include a relevant upcoming notice with its effective date labeled in the excerpt. It uses simple term overlap to select relevant paragraphs; policy paragraphs outrank FAQ summaries. The workflow uses the retrieval result to force `no_answer` and empty citations when no applicable paragraph matches, and suggests emailing the support team.

## Returns, warranties, and cancellation

Return eligibility runs in Python, reading `return_window_days` from SQLite. Day zero is delivery and the last configured day is included. Product return restrictions, opened-item rules, order ownership, item membership, delivery, and reason are checked before proposing a return. Warranty end dates use the configured standard or extended month count and calendar-month arithmetic (`relativedelta`); the end date is covered.

The customer first gets a proposal that ends with "Shall I go ahead?", `pending_confirmation: true`, and a `create_return_request` or `create_support_ticket` record with status `pending_confirmation` in `tool_calls`. A return proposal runs `get_orders` (filtered to the named order), `search_policy` ("return window"), and `check_return_eligibility` first, states the days since delivery and the configured window, and cites the return-window clause (`returns_policy` §2.1). A question such as "Can I return it? I changed my mind." only gets an eligibility answer and an offer. No database row is inserted on that turn. A clear affirmative on the very next message executes the proposed action after revalidation; any other next message discards it. Session requests are serialized, and a repeated affirmative after a completed action cannot create a duplicate row. An imperative such as `just do it` is not confirmation. Customer dates without a year are interpreted as dates in 2026. Return requests use status `requested`; support tickets use status `open`.

## Privacy and assumptions

Every order lookup is scoped to the customer header; unknown and other-customer orders have the same `ORDER_NOT_FOUND` response, including when a return request names an order that is not on the account (route `no_answer`, "I couldn't find that order on your account."). A session is tied to its first customer and a different customer receives HTTP 403. Missing or unknown customer IDs receive HTTP 401; invalid request JSON or required fields receive HTTP 400. Order-not-found is a normal HTTP 200 outcome.

Sessions are held in process memory, so they reset on restart and are not shared across multiple server workers. Use one worker for this assessment. The API key is optional for startup; without it only deterministic order lookup and excerpt-based policy fallback work, while the full natural-language agent and confirmation proposal workflow require OpenAI. The default model, `gpt-6-luna`, supports Chat Completions function calling when reasoning is disabled; ResolveBot sends `reasoning_effort="none"` and `temperature=0`. The request workflow has a 27-second processing timeout to leave time for HTTP handling within the 30-second requirement. The supplied source data and schema are not changed.

The implementation was exercised through real HTTP requests with the configured OpenAI model and a temporary copy of the SQLite database. Verification covered order lookup, the response schema, access isolation, confirmation-only writes, policy coverage routing, and timeout handling. The LangChain Core Pydantic v1 compatibility warning appears on Python 3.14 but did not prevent the app from starting or processing requests.

## Example chats

These examples were checked against the supplied data on `APP_TODAY=2026-07-01`:

| Customer message | Verified result |
| --- | --- |
| `C1009`: “Where is order SS-10512?” | “Order SS-10512 is in transit. The estimated delivery date is 2026-07-03.” The `get_orders` call succeeded. |
| `C1001`: “Where is order SS-10512?” | “I couldn't find that order on your account.” The HTTP response was 200, route was `no_answer`, and `get_orders` recorded an error. |
| “What is the standard shipping fee?” | The current policy excerpt returns ₹99 per order and cites `shipping_policy`, section `§2.1`. A future-dated policy is excluded from this current-policy query. |

Return-window boundaries and changed policy parameters were checked using a temporary database copy. The live OpenAI/LangGraph path was exercised over HTTP; tool selection and wording remain model-dependent.

## AI coding assistants

Codex was used to inspect the supplied SQLite database and policy files, implement the API, and prepare local verification. Review the deterministic rules, tool boundary, and assumptions before extending the project. `CLAUDE.md` provides concise rules for future AI-assisted coding.
