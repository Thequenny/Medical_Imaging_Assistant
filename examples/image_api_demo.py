import base64

from src.vlm.api_config import (
    QWEN_MODEL,
    create_qwen_client,
)


client = create_qwen_client()
with open("examples/assets/image.png", "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")
    #image= base64.b64encode(f.read()).decode()
task=[{ "type": "text",
        "text": "give me the fields in this invoice ."},
    {"type": "image_url",
     "image_url": {"url": f"data:image/png;base64,{image}"}
    }]
#print(task)
resp = client.chat.completions.create(
    model=QWEN_MODEL,
    messages=[
        {"role": "user", "content": task}
    ],
)
print(resp.choices[0].message.content)
