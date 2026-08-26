from pathlib import Path
import base64
import nibabel as nib
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas

from src.test_accuracy_evaluation.api_config import (
    QWEN_MODEL,
    create_qwen_client,
)


image = nib.load("dataset/serie_003.nii.gz")

def convert_to_png(image):
    data = image.get_fdata()

    # Save of slice images
    output_dir = Path("dataset/extracted_slices")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset old images
    for old_image in output_dir.glob("*.png"):
        old_image.unlink()

    # course of axial slice
    # sagittal=x, coronal=y and axial=z
    for z in range(data.shape[2]):
        slice_2d = data[:, :, z]
        # Normalisation simple between 0 and 255c
        slice_2d = slice_2d - np.min(slice_2d)
        if np.max(slice_2d) != 0:
            slice_2d = slice_2d / np.max(slice_2d)
        slice_2d = (slice_2d * 255).astype(np.uint8)
        # conversion to .png file
        slice_image = Image.fromarray(slice_2d)
        slice_image.save(output_dir / f"slice_{z:03d}.png")


    return output_dir, data.shape[2]


output_dir, number_of_slices = convert_to_png(image)


 # mini chat (questions-answers)
def chat_qwen(output_dir, number_of_slices):
    client = create_qwen_client()
    messages = []
    slices = np.arange(0, number_of_slices)

    while True:
        print("\n")
        prompt = input("prompt > ")
        if prompt == "end":
            break

        task = [{"type": "text", "text": prompt}]

        for i in slices:
            image_path = output_dir / f"slice_{i:03d}.png"

            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            task.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            })

        messages.append({"role": "user","content": task})

        resp = client.chat.completions.create(
              model=QWEN_MODEL,
              messages=messages,
          )
        answer = resp.choices[0].message.content
        print("\n")
        print(" Qwen > ", answer)

        messages.append({"role": "assistant","content": answer})

def generate_pdf(text):
    c = canvas.Canvas('slice_report.pdf')
    c.drawString(100, 800, text)
    c.save()


chat_qwen(output_dir, number_of_slices)


#print(resp.choices[0].message.content)
