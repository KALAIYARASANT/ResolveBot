# ResolveBot project instructions

- Keep the required stack: FastAPI, LangGraph, `langchain-openai`, Pydantic, built-in `sqlite3`, and `python-dateutil`.
- Keep one agent and exactly five customer support tools. Do not add endpoints or SQLite tables.
- Customer identity comes only from `X-Customer-Id`; every order operation must scope by that ID.
- Eligibility and warranty calculations belong in deterministic Python and read current parameters from `policy_parameters`.
- Treat `APP_TODAY` as the only business date. Do not alter supplied data or database schema.
- Never perform return or ticket writes before explicit confirmation in the same session. Revalidate directly before insertion.
- Policy content comes from the configured policy file and manifest. Filter by status and effective date and cite excerpts actually used.
- Keep API responses within the documented schema and map malformed requests to HTTP 400 rather than FastAPI 422.
- Keep changes small, explain assumptions, and update README examples when behavior changes.
