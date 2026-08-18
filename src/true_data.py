import base64
import json

import nibabel as ni
import numpy as np
from huggingface_hub import snapshot_download
from openai import OpenAI
from slices_analyse import convert_to_png


client = OpenAI(
    base_url="https://spark-da32.tail67be05.ts.net:8443/v1",
    api_key="9f16632ff4b7a61eea6c1a9aa8f37464b9d2f795395ac45e",
    timeout=300.0,
)

# Download the dataset where the paths from train.jsonl expect it.
snapshot_download(
    repo_id="Angelou0516/chaos-mri",
    repo_type="dataset",
    local_dir="data/nii/CHAOS",
)

data = []
with open("src/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Access a sample from train.jsonl.
sample = data[3]
#print(sample)
#print(f"Patient ID: {sample['patient_id']}")
#print(f"Image: {sample['image']}")
#print(f"Mask: {sample['mask']}")
#print(f"Labels: {sample['label']}")

mask_path = sample["mask"]
mask_nii = ni.load(mask_path)
mask = mask_nii.get_fdata()

# creation of the "trues data" to evaluate the model

def test_true(data):

    test={"organs": [], "pixel": {} , "vertebra": ""}
    for i in range(data):
        patient=data[i]
        test['organs']=patient['label']
        test["vertebra"]=patient['modality'].replace("MRI ","")
        test["pixel"]={pixel: int(np.count_nonzero(mask[j]))
                       for j,pixel in enumerate(patient['label'])}
        
        with open("dice.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(test) + "\n")


prompt="""
Tu vas recevoir toutes les coupes axiales d'un même volume IRM abdominal,
classées dans leur ordre d'origine.

Analyse l'ensemble du volume et réponds uniquement avec un objet JSON valide,
sur une seule ligne, sans Markdown, sans commentaire et sans bloc ```json.

Le format attendu est exactement :
{"organs": [], "pixel": {}, "vertebra": ""}

Explication: :
- "organs" contient uniquement les organes visibles parmi : liver,
  right kidney, left kidney et spleen ;
- "pixel" associe chaque organe visible au nombre total estimé de pixels qu'il
  occupe sur l'ensemble des coupes, sous la forme d'un entier positif ;
- pour rester compatible avec le JSON de référence, "vertebra" contient le
  type de séquence IRM détecté : "T1" ou "T2" ;
- utilise exactement les noms de champs et d'organes indiqués ci-dessus.
"""

output_path = "src/all_slices.jsonl"

try:
    with open(output_path, "r", encoding="utf-8") as output_file:
        completed = sum(1 for line in output_file if line.strip())
except FileNotFoundError:
    completed = 0

print(f"Reprise à l'examen {completed + 1}/{len(data)}")

with open(output_path, "a", encoding="utf-8") as output_file:
    for patient_number, patient in enumerate(
        data[completed:],
        start=completed + 1,
    ):
        print(
            f"[{patient_number}/{len(data)}] Conversion des coupes du patient "
            f"{patient['patient_id']}...",
            flush=True,
        )
        output_dir, number_of_slices = convert_to_png(
            "axial slice",
            patient["image"],
        )

        task = [{"type": "text", "text": prompt}]

        for i in range(number_of_slices):
            image_path = output_dir / f"slice_{i:03d}.png"
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

            task.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            })

        print(
            f"[{patient_number}/{len(data)}] Envoi de {number_of_slices} coupes "
            "à Qwen...",
            flush=True,
        )
        resp = client.chat.completions.create(
            model="qwen3.6:35b",
            messages=[
                {"role": "user", "content": task}
            ],
        )

        answer = json.loads(resp.choices[0].message.content)
        output_file.write(json.dumps(answer) + "\n")
        output_file.flush()
        print(
            f"[{patient_number}/{len(data)}] Réponse enregistrée dans {output_path}",
            flush=True,
        )


def calculate_score(true_path, prediction_path):
    with open(true_path, "r", encoding="utf-8") as file:
        true_data = [json.loads(line) for line in file if line.strip()]

    with open(prediction_path, "r", encoding="utf-8") as file:
        predictions = [json.loads(line) for line in file if line.strip()]

    if not predictions:
        raise ValueError("Aucune prédiction à évaluer.")

    organ_scores = []
    pixel_scores = []
    vertebra_scores = []

    for true, prediction in zip(true_data, predictions):
        true_organs = set(true["organs"])
        predicted_organs = set(prediction.get("organs", []))

        organ_scores.append(
            2 * len(true_organs & predicted_organs)
            / (len(true_organs) + len(predicted_organs))
        )

        organ_pixel_scores = []
        for organ, true_pixels in true["pixel"].items():
            predicted_pixels = prediction.get("pixel", {}).get(organ, 0)
            total_pixels = true_pixels + predicted_pixels
            score = (
                2 * min(true_pixels, predicted_pixels) / total_pixels
                if total_pixels > 0
                else 1
            )
            organ_pixel_scores.append(score)

        pixel_scores.append(np.mean(organ_pixel_scores))
        vertebra_scores.append(
            true["vertebra"] == prediction.get("vertebra", "")
        )

    organ_score = float(np.mean(organ_scores))
    pixel_score = float(np.mean(pixel_scores))
    vertebra_score = float(np.mean(vertebra_scores))

    return {
        "score_organes (en %)": round(organ_score * 100, 2),
        "score_pixels (en %) ": round(pixel_score * 100, 2),
        "score_vertebra (en %)" : round(vertebra_score * 100, 2),
        "score_global (en %)": round(
            np.mean([organ_score, pixel_score, vertebra_score]) * 100,
            2,
        ),
    }


scores = calculate_score("src/dice.jsonl", output_path)
print(json.dumps(scores, indent=2, ensure_ascii=False))
