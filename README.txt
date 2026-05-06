Muse Spark CLI + Local OpenAI-Compatible API

What this is
- A local CLI and FastAPI wrapper for Meta AI Muse Spark.
- It stores auth locally so day to day usage is just `new`, `chat`, `use`, `current`, and `list`.
- It opens the captured `gateway.meta.ai` WebSocket.
- It uses the home prompt template for new conversations.
- It uses the chat prompt template for follow ups and resumed conversations.
- It now exposes a local OpenAI-compatible `/v1/chat/completions` endpoint for your own agents and tools.

What it is not
- Not a public API client.
- Not a real login flow yet.
- Still reverse engineered, so Meta can break it whenever they feel like being annoying.
- Streaming is now true token-level SSE (as received from Meta), no longer artificially chunked by default.

Project layout
- `muse_spark/client.py`: protobuf patcher, session store, HTTP helpers, CLI, protocol transport
- `muse_spark/provider.py`: provider adapter over the transport layer
- `muse_spark/prompt_compiler.py`: OpenAI-style messages -> Muse prompt compiler
- `muse_spark/api.py`: FastAPI app factory and HTTP routes
- `muse_spark/openai_compat.py`: OpenAI-style response and SSE helpers
- `muse_spark/schemas.py`: request schemas
- `muse_spark/config.py`: env-driven API settings
- `muse_spark/logging_utils.py`: logger setup
- `tests/`: regression and API tests
- `pyproject.toml`: project metadata and runtime dependencies
- `requirements.txt`: minimal compatibility fallback dependency list
- `.env.example`: optional environment variable examples

Setup with uv (recommended)
1. `cd /Users/kamell/Documents/Projects/labs/muse-spark`
2. `uv sync`
3. Run commands with `uv run ...`

Examples:
- `uv run muse-spark --help`
- `uv run python -m unittest discover -s tests -v`

Fallback setup without uv
1. `cd /Users/kamell/Documents/Projects/labs/muse-spark`
2. `python3 -m venv .venv`
3. `source .venv/bin/activate`
4. `pip3 install -r requirements.txt`

One time auth setup
You still need both values from Charles for now:
- Cookie header from Meta requests
- `ecto1:...` authorization token from the WebSocket query string

Store them once:
`uv run muse-spark auth set --cookie 'datr=...; ecto_1_sess=...; ...' --authorization 'ecto1:...'`

CLI usage
Start a new conversation:
`uv run muse-spark new "new convo probe 1"`

Send to the current conversation:
`uv run muse-spark chat "follow up probe 2"`

Switch current conversation:
`uv run muse-spark use b08385a6-5a53-4f14-966e-347f28088454`

Show current conversation:
`uv run muse-spark current`

List known conversations:
`uv run muse-spark list`

Debug a generated frame:
`uv run muse-spark debug-frame "hello world" --conversation-id 0408aded-55f9-4748-bcdf-dfe5f13b337b --template home`

Run the local API
Start the server:
`uv run muse-spark serve --host 127.0.0.1 --port 8000`

Force a single persistent conversation for AI agents:
`uv run muse-spark serve --single-conversation`

Adjust streaming and timeouts:
`uv run muse-spark serve --chunk-size 0 --timeout 15.0`

Health check:
`curl http://127.0.0.1:8000/healthz`

List models:
`curl http://127.0.0.1:8000/v1/models`

Non-streaming completion:
`curl http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"meta/muse-spark","messages":[{"role":"user","content":"Refactor this React navbar component."}]}'`

Streaming completion:
`curl -N http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"meta/muse-spark","stream":true,"messages":[{"role":"user","content":"Reply with exactly: pong"}]}'`

Environment variable fallback
The CLI/API prefer values stored in `~/.muse_spark/state.json`, but they can also fall back to environment variables:
- `MUSE_SPARK_COOKIE_HEADER`
- `MUSE_SPARK_COOKIE`
- `MUSE_SPARK_AUTHORIZATION`
- `MUSE_SPARK_MODE`

API settings from env
- `MUSE_SPARK_MODEL_NAME` default: `meta/muse-spark`
- `MUSE_SPARK_LOG_LEVEL` default: `INFO`
- `MUSE_SPARK_STREAM_CHUNK_SIZE` default: `0` (0 to disable artificial chunking)
- `MUSE_SPARK_DEBUG_FRAME_DUMPS` default: `0`
- `MUSE_SPARK_FORCE_SINGLE_CONVERSATION` default: `0` (Set to 1 to stick to one conversation)
- `MUSE_SPARK_RECEIVE_TIMEOUT` default: `10.0`

Notes
- The API is stateful: each OpenAI conversation_id maps to a persistent Meta conversation under the hood.
- Use `--single-conversation` if you want all requests to share the exact same history (useful for simple agents).
- New conversations use a hidden bootstrap turn, then the real user turn.
- Follow ups reuse the same conversation_id and send only the latest user message.
- The stateless transcript compiler is removed; stateful XML turn planning is the only chat path.
- The CLI stores known conversations locally in `~/.muse_spark/state.json`.
- If auth expires, rerun `auth set` with fresh Charles values.
- `response_format={"type":"json_object"}` is best-effort prompting, not hard schema enforcement.
- `max_tokens` and `stop` are advisory prompt guidance for now, not provider-native controls.
- A proper login command comes later.
