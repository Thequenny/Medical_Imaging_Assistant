from openai import OpenAI
import base64
client = OpenAI(
    base_url="https://spark-da32.tail67be05.ts.net:8443/v1",
    api_key="9f16632ff4b7a61eea6c1a9aa8f37464b9d2f795395ac45e",
)
with open("src/image.png", "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")
    #image= base64.b64encode(f.read()).decode()
task=[{ "type": "text",
        "text": "give me the fields in this invoice ."},
    {"type": "image_url",
     "image_url": {"url": f"data:image/png;base64,{image}"}
    }]
#print(task)
resp = client.chat.completions.create(
    model="qwen3.6:35b",
    messages=[
        {"role": "user", "content": task}
    ],
)
print(resp.choices[0].message.content)
