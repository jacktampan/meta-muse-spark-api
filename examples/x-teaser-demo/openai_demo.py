from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="x",
)

resp = client.chat.completions.create(
    model="meta/muse-spark",
    messages=[{
        "role": "user",
        "content": "Reply with exactly: muse spark is live"
    }],
)

print(resp.choices[0].message.content)
