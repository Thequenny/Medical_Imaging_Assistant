import argparse
import base64
import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from openai import OpenAI


DEFAULT_MAX_CASES = 10
PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_DIR / "src" / "train.jsonl"
REFERENCE_PATH = PROJECT_DIR / "src" / "dice.jsonl"
PREDICTION_PATH = PROJECT_DIR / "src" / "all_slices.jsonl"
DATASET_DIR = PROJECT_DIR / "data" / "nii" / "CHAOS"

DATASET_REPO_ID = "Angelou0516/chaos-mri"
DATASET_PATH_PREFIX = Path("data/nii/CHAOS")

QWEN_BASE_URL = "https://spark-da32.tail67be05.ts.net:8443/v1"
QWEN_API_KEY = "9f16632ff4b7a61eea6c1a9aa8f37464b9d2f795395ac45e"
QWEN_MODEL = "qwen3.6:35b"


PROMPT = """
Tu vas recevoir toutes les coupes axiales d'un même volume IRM abdominal,
classées dans leur ordre d'origine.

Analyse l'ensemble du volume et réponds uniquement avec un objet JSON valide,
sur une seule ligne, sans Markdown, sans commentaire et sans bloc ```json.

Le format attendu est exactement :
{"organs": [], "pixel": {}, "vertebra": ""}

Explication :
- "organs" contient uniquement les organes visibles parmi : liver,
  right kidney, left kidney et spleen ;
- "pixel" associe chaque organe visible au nombre total estimé de pixels qu'il
  occupe sur l'ensemble des coupes, sous la forme d'un entier positif ;
- pour rester compatible avec le JSON de référence, "vertebra" contient le
  type de séquence IRM détecté : "T1" ou "T2" ;
- utilise exactement les noms de champs et d'organes indiqués ci-dessus.
"""


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_cases(max_cases):
    if max_cases <= 0:
        raise ValueError("Le nombre d'examens doit être strictement positif.")

    cases = load_jsonl(TRAIN_PATH)
    if len(cases) < max_cases:
        raise ValueError(
            f"Seulement {len(cases)} examens sont disponibles dans {TRAIN_PATH}."
        )

    return cases[:max_cases]


def dataset_relative_path(path):
    try:
        return Path(path).relative_to(DATASET_PATH_PREFIX).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Le chemin {path!r} ne se trouve pas sous {DATASET_PATH_PREFIX}."
        ) from error


def download_dataset(cases, download_all=False):
    if download_all:
        allow_patterns = None
        print("Téléchargement du dataset CHAOS MRI complet...")
    else:
        allow_patterns = sorted(
            {
                dataset_relative_path(case[path_type])
                for case in cases
                for path_type in ("image", "mask")
            }
        )
        print(
            "Téléchargement limité aux fichiers nécessaires aux "
            f"{len(cases)} examens..."
        )

    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=DATASET_DIR,
        allow_patterns=allow_patterns,
    )


def load_existing_predictions(prediction_path, max_cases):
    prediction_path = Path(prediction_path)
    if not prediction_path.exists():
        return []

    predictions = load_jsonl(prediction_path)
    if len(predictions) > max_cases:
        print(
            f"{len(predictions) - max_cases} ancienne(s) prédiction(s) "
            f"supplémentaire(s) seront ignorée(s)."
        )

    return predictions[:max_cases]


def generate_predictions(cases, prediction_path=PREDICTION_PATH):
    from slices_analyse import convert_to_png

    prediction_path = Path(prediction_path)
    predictions = load_existing_predictions(prediction_path, len(cases))
    completed = len(predictions)

    if completed == len(cases):
        print(f"Les {len(cases)} examens ont déjà été analysés.")
        return

    print(f"Reprise à l'examen {completed + 1}/{len(cases)}")

    client = OpenAI(
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
        timeout=300.0,
    )

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("a", encoding="utf-8") as output_file:
        for patient_number, patient in enumerate(
            cases[completed:],
            start=completed + 1,
        ):
            print(
                f"[{patient_number}/{len(cases)}] Conversion des coupes de "
                f"l'examen {patient['patient_id']}...",
                flush=True,
            )
            image_path = PROJECT_DIR / patient["image"]
            output_dir, number_of_slices = convert_to_png(
                "axial slice",
                image_path,
                root_dir=PROJECT_DIR,
            )

            task = [{"type": "text", "text": PROMPT}]
            for slice_number in range(number_of_slices):
                slice_path = output_dir / f"slice_{slice_number:03d}.png"
                with slice_path.open("rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode(
                        "utf-8"
                    )

                task.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    }
                )

            print(
                f"[{patient_number}/{len(cases)}] Envoi de "
                f"{number_of_slices} coupes à Qwen...",
                flush=True,
            )
            response = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": task}],
            )

            answer = json.loads(response.choices[0].message.content)
            output_file.write(json.dumps(answer, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{patient_number}/{len(cases)}] Réponse enregistrée dans "
                f"{prediction_path}",
                flush=True,
            )


def calculate_score(true_path, prediction_path, max_cases=DEFAULT_MAX_CASES):
    true_data = load_jsonl(true_path)[:max_cases]
    predictions = load_jsonl(prediction_path)[:max_cases]

    if len(true_data) != max_cases:
        raise ValueError(
            f"{max_cases} références sont requises, mais {len(true_data)} "
            "seulement sont disponibles."
        )
    if len(predictions) != max_cases:
        raise ValueError(
            f"{max_cases} prédictions sont requises, mais {len(predictions)} "
            "seulement sont disponibles."
        )

    organ_scores = []
    pixel_scores = []
    sequence_scores = []

    for true, prediction in zip(true_data, predictions):
        true_organs = set(true["organs"])
        predicted_organs = set(prediction.get("organs", []))
        organ_denominator = len(true_organs) + len(predicted_organs)
        organ_scores.append(
            2 * len(true_organs & predicted_organs) / organ_denominator
            if organ_denominator
            else 1.0
        )

        organ_pixel_scores = []
        for organ, true_pixels in true["pixel"].items():
            predicted_pixels = prediction.get("pixel", {}).get(organ, 0)
            if not isinstance(predicted_pixels, (int, float)):
                predicted_pixels = 0

            predicted_pixels = max(predicted_pixels, 0)
            total_pixels = true_pixels + predicted_pixels
            organ_pixel_scores.append(
                2 * min(true_pixels, predicted_pixels) / total_pixels
                if total_pixels > 0
                else 1.0
            )

        pixel_scores.append(
            float(np.mean(organ_pixel_scores)) if organ_pixel_scores else 1.0
        )
        sequence_scores.append(
            true["vertebra"] == prediction.get("vertebra", "")
        )

    organ_score = float(np.mean(organ_scores))
    pixel_score = float(np.mean(pixel_scores))
    sequence_score = float(np.mean(sequence_scores))

    return {
        "examens_evalues": max_cases,
        "score_organes (en %)": round(organ_score * 100, 2),
        "score_pixels (en %)": round(pixel_score * 100, 2),
        "score_sequence_IRM (en %)": round(sequence_score * 100, 2),
        "score_global (en %)": round(
            np.mean([organ_score, pixel_score, sequence_score]) * 100,
            2,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Évalue Qwen sur un petit échantillon du dataset CHAOS MRI."
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Télécharger le dataset complet au lieu des fichiers de l'échantillon.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=DEFAULT_MAX_CASES,
        help=f"Nombre d'examens à analyser (défaut : {DEFAULT_MAX_CASES}).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Calculer le score existant sans téléchargement ni appel à Qwen.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = load_cases(args.max_cases)

    if not args.score_only:
        download_dataset(cases, download_all=args.download_all)
        generate_predictions(cases)

    scores = calculate_score(
        REFERENCE_PATH,
        PREDICTION_PATH,
        max_cases=len(cases),
    )
    print(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
