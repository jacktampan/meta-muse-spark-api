Muse Spark X teaser demo

Use `openai_demo.py` for the recording.
It is intentionally tiny and camera friendly.

Run it
`uv run --with openai python examples/x-teaser-demo/openai_demo.py`

What to show on screen
- the `from openai import OpenAI` import
- `base_url="http://127.0.0.1:8000/v1"`
- `model="meta/muse-spark"`
- the prompt line
- the terminal output: `muse spark is live`

If you want a more configurable local helper with flags and optional streaming, use the internal module instead:
`uv run python -m muse_spark.demo_openai_client`

Recording advice
- keep the file already open
- show the whole file in one shot
- run it from the IDE terminal
- cut as soon as the output lands

Don't overcook it. The point is that existing OpenAI SDK code works unchanged except for `base_url`.
