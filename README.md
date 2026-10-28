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
# Edit .env and set OPENAI_API_KEY (and OPENAI_MODEL if your key cannot use the default)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `POLICIES_PATH`, `MANIFEST_PATH`, `DB_PATH` | `./data/...` | Policy file, its manifest, and the SQLite database. |
| `APP_TODAY` | `2026-07-01` | The only business date (never the computer's clock). Also used for the date part of saved timestamps. |
| `LLM_PROVIDER` | `openai` | Only `openai` is supported. Any other value stops startup with `ConfigurationError: LLM_PROVIDER='…' is not supported`; there is no silent fallback. |
| `OPENAI_API_KEY` | empty | Without it the service still starts (with a warning) but only the limited deterministic fallback works. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any chat model your key can use that supports function calling and `temperature=0`. An empty value is a startup error. |
| `OPENAI_REASONING_EFFORT` | empty | Sent only if set. Reasoning models that reject `temperature=0` need it, for example `OPENAI_MODEL=gpt-6-luna` with `OPENAI_REASONING_EFFORT=none`. |

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
- `app/core/config.py` loads and validates environment configuration; `app/core/ids.py` normalizes order IDs.
- `app/agents/support_agent.py` runs one LangGraph model/tool loop, with a six-call maximum and session handling.
- `app/tools/support_tools.py` defines the five tools and binds customer identity from trusted Python context.
- `app/services/business_rules.py` implements return and warranty decisions and revalidates writes.
- `app/services/policy_search.py` reads policy documents and manifest entries, then retrieves relevant current paragraphs.
- `app/db/database.py` contains small SQLite connection and lookup helpers.

The tools are `get_orders(order_id=None, status=None)`, `search_policy(query)`, `check_return_eligibility(order_id, item_ids, reason)`, `create_return_request(order_id, item_ids, reason)`, and `create_support_ticket(order_id, ticket_type, summary)`. The model chooses tools; Python checks all authority-sensitive and eligibility-sensitive data. The customer ID is never a model argument.

## Policy retrieval and citations

The retriever splits the configured Markdown file by document ID and numbered headings, keeping each section's heading (FAQ questions live in the headings) and the manifest title. It matches each document against `manifest.json`, excludes superseded policies, and excludes future-effective notices from current-policy questions using `APP_TODAY`; a question that mentions a future date or change can include an upcoming notice, labeled with its effective date. Matching is stemmed term overlap with no topic-to-document map: a section needs at least half of a short question's terms (at most two for long queries) to count, so one shared word is not coverage, and every document, FAQ included, competes on the same score (policy wins ties). When two in-force policy sections cover the same topic (same heading and matched terms), the newer document replaces the older one (for example the shipping fee once the update is in force).

The model sees at most five excerpts (two per document) and the business date. It ends its reply with a `Sources:` line, which is stripped from the answer. The cited sections are the ones it names that also appear in its answer, plus the one retrieved excerpt the answer overlaps most; an answer that does not use an excerpt cites nothing. No coverage returns `no_answer`, empty citations, and the "policies do not cover that question, email the support team" reply. Questions that are not about ShopSphere (code, sports, weather, ...) return `out_of_scope` with empty `tool_calls` and `citations`. If the model refuses or says "not covered" without searching while the policy files do match the message, the agent gets one retry told to search.

## Returns, warranties, and cancellation

Return eligibility runs in Python, reading `return_window_days` from SQLite. Day zero is delivery and the last configured day is included. Product return restrictions, opened-item rules, order ownership, item membership, delivery, and reason are checked before proposing a return. Warranty end dates use the configured standard or extended month count and calendar-month arithmetic (`relativedelta`); the end date is covered.

Order and item IDs are normalized before any lookup (`SS10512`, `SS 10512` and `ss-10512` all mean `SS-10512`); normalization never widens access, because every lookup still filters by the authenticated customer. Python also checks every order ID the customer named: if none is theirs the reply is exactly "I couldn't find that order on your account." (`no_answer`); if the message also asked something answerable, that part is answered and the sentence is added.

A message that asks several things (several order IDs, tracking plus a return, or more than one question) skips the scripted return shortcuts and goes to the agent loop, so each part is answered with its own tool calls. A follow-up such as "When will it arrive?" is resolved from the previous turns when they named one order; without such context the bot lists the customer's orders and asks which one. Warranty questions use `get_orders` (name, category, `warranty_active`, `warranty_end_date`) to find the item and `search_policy` for what is covered; a ticket is offered but only proposed when the customer asks. Write tools are not offered to the model on turns where the customer did not ask for an action.

A second return request for an item that already has one (or a second cancellation ticket for an order that already has an open one) is refused with `ALREADY_REQUESTED`, both when proposing and again at insertion, inside one `BEGIN IMMEDIATE` transaction. `check_return_eligibility` itself is unchanged and answers only the business rules. Warranty tickets are not de-duplicated, because separate faults on one order are legitimate.

The customer first gets a proposal that ends with "Shall I go ahead?", `pending_confirmation: true`, and a `create_return_request` or `create_support_ticket` record with status `pending_confirmation` in `tool_calls`. A return proposal runs `get_orders` (filtered to the named order), `search_policy` ("return window"), and `check_return_eligibility` first, states the days since delivery and the configured window, and cites the return-window clause (`returns_policy` §2.1). A question such as "Can I return it? I changed my mind." only gets an eligibility answer and an offer. No database row is inserted on that turn. A clear affirmative on the very next message executes the proposed action after revalidation; any other next message discards it. Session requests are serialized, and a repeated affirmative after a completed action cannot create a duplicate row. An imperative such as `just do it` is not confirmation. Customer dates without a year are interpreted as dates in 2026. Return requests use status `requested`; support tickets use status `open`.

## Privacy and assumptions

Every order lookup is scoped to the customer header; unknown and other-customer orders have the same `ORDER_NOT_FOUND` response, including when a return request names an order that is not on the account (route `no_answer`, "I couldn't find that order on your account."). A session is tied to its first customer and a different customer receives HTTP 403. Missing or unknown customer IDs receive HTTP 401; invalid request JSON or required fields receive HTTP 400. Order-not-found is a normal HTTP 200 outcome.

Sessions are held in process memory, so they reset on restart and are not shared across multiple server workers. Use one worker for this assessment. The API key is optional for startup; without it only deterministic order lookup and excerpt-based policy fallback work, while the full natural-language agent and confirmation proposal workflow require OpenAI. The model comes from `OPENAI_MODEL` (default `gpt-4o-mini`) and is always called with `temperature=0`; `reasoning_effort` is sent only when `OPENAI_REASONING_EFFORT` is set. The request workflow has a 27-second processing timeout to leave time for HTTP handling within the 30-second requirement. The supplied source data and schema are not changed.

Assumptions and limits: a `message` is at most 4000 characters and a `session_id` at most 128 (longer is HTTP 400). Idle sessions are forgotten after an hour, at most 1000 are kept (least recently used go first), and only the last 40 messages of a chat are stored, so a proposal waiting in an expired session is lost and a later "yes" saves nothing. Saved `created_at` values use the `APP_TODAY` date with the wall-clock time of day, since `APP_TODAY` is the only business date. The shipped model prompt contains no policy facts; the business date and the rules for choosing between excerpts are the only domain text it carries.

The implementation was exercised through real HTTP requests with the configured OpenAI model and a temporary copy of the SQLite database. Verification covered order lookup, the response schema, access isolation, confirmation-only writes, policy coverage routing, and timeout handling. The LangChain Core Pydantic v1 compatibility warning appears on Python 3.14 but did not prevent the app from starting or processing requests.

## Example chats

These examples were checked against the supplied data on `APP_TODAY=2026-07-01`:

| Customer message | Verified result |
| --- | --- |
| `C1009`: “Where is order SS-10512?” | “Order SS-10512 is in transit. The estimated delivery date is 2026-07-03.” The `get_orders` call succeeded. |
| `C1001`: “Where is order SS-10512?” | “I couldn't find that order on your account.” The HTTP response was 200, route was `no_answer`, and `get_orders` recorded an error. |
| “What is the standard shipping fee?” | The current policy excerpt returns ₹99 per order and cites `shipping_policy`, section `§2.1`. A future-dated policy is excluded from this current-policy query. With `APP_TODAY=2026-08-02` the in-force update applies instead: ₹149, citing `shipping_policy_update` `§1.1`. |
| `C1009`: “Where is SS-10512, and can I return the speaker from SS-10458?” | Both parts answered in one reply: in transit, estimated delivery 3 July 2026; the Nimbra Smart Speaker is eligible if they changed their mind, and the bot offers to start the return (nothing saved). |
| `C1001`: “Do you ship to Dubai?” | No, ShopSphere delivers only within India. Cites `faq_general` `§6.1`. |
| `C1001`: “My phone stopped charging. Is it covered?” | The Nimbra One is under warranty until 14 February 2027 and charging faults are covered; cites `warranty_policy` `§3.1` and offers a warranty ticket without proposing one. |
| `C1001`: “Do you price match competitors?” | Route `no_answer`, no citations, "ShopSphere's policies do not cover that question. Please email the support team for help." |
| `C1001`: “Write python code to reverse a list” | Route `out_of_scope`, empty `tool_calls` and `citations`. |
| `C1009`: “Where is order SS-10512?” then “When will it arrive?” | The second answer resolves "it" to SS-10512 (3 July 2026). |
| `C1009`: return the speaker twice | The second request is refused: "A return request already exists for one of these items." |

Return-window boundaries and changed policy parameters (window, warranty months, policy text, manifest dates, `APP_TODAY`) were checked using temporary database copies. The examples above were produced by the live OpenAI path with `gpt-4o-mini`; tool selection and wording remain model-dependent, so wording can differ between runs and models.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest tests.test_scenarios tests.test_interview_fixes
```

Both files use temporary copies of the database and no network (a scripted stand-in replaces the model). `tests/test_interview_fixes.py` holds the regression tests for multi-question messages, FAQ retrieval, warranty data, routing, citations, conversation context, ID normalization, duplicate prevention, session limits and configuration errors. The `tests/run_*_live.py` scripts call the real OpenAI API.

Known limitations: the agent's wording and tool choices depend on the model (the scripted return flow and all rules are deterministic); retrieval is lexical, so a question phrased with none of a section's words can be missed; the scripted return shortcuts still rely on regular expressions for recognising return requests and reasons; sessions live in one process only.

## AI coding assistants

Codex was used to inspect the supplied SQLite database and policy files, implement the API, and prepare local verification. Claude Code was later used to test the service against the assessment's five mandatory checks and extra scenarios, and to make the follow-up fixes and regression tests described above. Review the deterministic rules, tool boundary, and assumptions before extending the project. `CLAUDE.md` provides concise rules for future AI-assisted coding.
