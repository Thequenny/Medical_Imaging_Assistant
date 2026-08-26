import argparse
import base64
import json
from contextlib import ExitStack
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    from .api_config import QWEN_MODEL, create_qwen_client
except ImportError:
    from api_config import QWEN_MODEL, create_qwen_client


DEFAULT_PATIENTS_PER_SEQUENCE = 5
RESPONSE_COUNT = 3
SEQUENCES = ("T1", "T2")

PROJECT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent
TRAIN_PATH = EVALUATION_DIR / "train.jsonl"
REFERENCE_PATH = EVALUATION_DIR / "dice.jsonl"
PREDICTION_PATHS = {
    response_number: (
        EVALUATION_DIR / f"predictions_{response_number}.jsonl"
    )
    for response_number in range(1, RESPONSE_COUNT + 1)
}
DATASET_DIR = PROJECT_DIR / "data" / "nii" / "CHAOS"

DATASET_REPO_ID = "Angelou0516/chaos-mri"
DATASET_PATH_PREFIX = Path("data/nii/CHAOS")

PROMPT = """
Tu vas recevoir toutes les coupes axiales d'un même volume IRM abdominal,
classées dans leur ordre d'origine.

Cette requête est une évaluation indépendante. Réponds toujours aux deux mêmes
questions :
1. Quels organes sont visibles ?
2. La séquence IRM est-elle T1 ou T2 ?

Réponds uniquement avec un objet JSON valide, sur une seule ligne, sans
Markdown, sans commentaire et sans bloc ```json.

Le format attendu est exactement :
{"organs": [], "sequence": ""}

Explication :
- "organs" contient uniquement les organes visibles parmi : liver,
  right kidney, left kidney et spleen ;
- "sequence" contient le type de séquence IRM détecté : "T1" ou "T2" ;
- utilise exactement les noms de champs et d'organes indiqués ci-dessus.
"""


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def case_sequence(case):
    return case.get("modality", "").removeprefix("MRI ").strip().upper()


def load_balanced_case_groups(
    patients_per_sequence=DEFAULT_PATIENTS_PER_SEQUENCE,
):
    if patients_per_sequence <= 0:
        raise ValueError(
            "Le nombre de patients par séquence doit être strictement positif."
        )

    cases = load_jsonl(TRAIN_PATH)
    references = load_jsonl(REFERENCE_PATH)
    if len(cases) != len(references):
        raise ValueError(
            "Le nombre d'examens et le nombre de références ne correspondent "
            f"pas : {len(cases)} contre {len(references)}."
        )

    patient_order = []
    records_by_patient = {}

    for case, reference in zip(cases, references):
        sequence = case_sequence(case)
        if sequence not in SEQUENCES:
            continue

        reference_sequence = str(reference.get("vertebra", "")).upper()
        if reference_sequence != sequence:
            raise ValueError(
                "Séquence incohérente pour le patient "
                f"{case.get('patient_id')}: {sequence} dans {TRAIN_PATH}, "
                f"mais {reference_sequence or 'vide'} dans {REFERENCE_PATH}."
            )

        patient_id = str(case.get("patient_id"))
        if patient_id not in records_by_patient:
            patient_order.append(patient_id)
            records_by_patient[patient_id] = {}

        # CHAOS contient deux volumes T1 par patient. Le premier est conservé
        # afin que chaque patient ne soit présent qu'une fois dans le groupe T1.
        records_by_patient[patient_id].setdefault(
            sequence,
            {"case": case, "reference": reference},
        )

    eligible_patient_ids = [
        patient_id
        for patient_id in patient_order
        if all(
            sequence in records_by_patient[patient_id]
            for sequence in SEQUENCES
        )
    ]

    required_patient_count = patients_per_sequence * len(SEQUENCES)
    if len(eligible_patient_ids) < required_patient_count:
        raise ValueError(
            f"Seulement {len(eligible_patient_ids)} patients possèdent à la "
            f"fois une séquence T1 et une séquence T2 ; "
            f"{required_patient_count} sont requis pour former deux groupes "
            "sans patient commun."
        )

    patient_ids_by_sequence = {
        "T1": eligible_patient_ids[:patients_per_sequence],
        "T2": eligible_patient_ids[
            patients_per_sequence:required_patient_count
        ],
    }
    return {
        sequence: [
            records_by_patient[patient_id][sequence]
            for patient_id in patient_ids_by_sequence[sequence]
        ]
        for sequence in SEQUENCES
    }


def dataset_relative_path(path):
    try:
        return Path(path).relative_to(DATASET_PATH_PREFIX).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Le chemin {path!r} ne se trouve pas sous {DATASET_PATH_PREFIX}."
        ) from error


def download_dataset(case_groups, download_all=False):
    if download_all:
        allow_patterns = None
        print("Téléchargement du dataset CHAOS MRI complet...")
    else:
        allow_patterns = sorted(
            {
                dataset_relative_path(record["case"]["image"])
                for records in case_groups.values()
                for record in records
            }
        )
        patient_count = sum(len(records) for records in case_groups.values())
        print(
            "Téléchargement limité aux images nécessaires aux "
            f"{patient_count} analyses..."
        )

    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=DATASET_DIR,
        allow_patterns=allow_patterns,
    )


def ordered_case_records(case_groups):
    return [
        record
        for sequence in SEQUENCES
        for record in case_groups[sequence]
    ]


def load_existing_predictions(prediction_path, records):
    prediction_path = Path(prediction_path)
    if not prediction_path.exists():
        return []

    predictions = load_jsonl(prediction_path)
    if len(predictions) > len(records):
        raise ValueError(
            f"{prediction_path} contient {len(predictions)} prédictions, "
            f"mais {len(records)} seulement sont attendues."
        )

    for record, prediction in zip(records, predictions):
        expected_patient_id = str(record["case"]["patient_id"])
        prediction_patient_id = str(prediction.get("patient_id", ""))
        if prediction_patient_id != expected_patient_id:
            raise ValueError(
                f"Patient inattendu dans {prediction_path}: "
                f"{prediction_patient_id or 'identifiant absent'} au lieu de "
                f"{expected_patient_id}."
            )

    return predictions


def normalize_prediction(answer, patient_id):
    if not isinstance(answer, dict):
        raise ValueError("La réponse de Qwen doit être un objet JSON.")

    organs = answer.get("organs")
    if not isinstance(organs, list) or not all(
        isinstance(organ, str) for organ in organs
    ):
        raise ValueError("Le champ 'organs' doit être une liste de textes.")

    sequence = answer.get("sequence", answer.get("vertebra", ""))
    if not isinstance(sequence, str):
        raise ValueError("Le champ 'sequence' doit être un texte.")

    return {
        "patient_id": patient_id,
        "organs": organs,
        "sequence": sequence.strip().upper(),
    }


def build_model_task(output_dir, number_of_slices):
    task = [{"type": "text", "text": PROMPT}]

    for slice_number in range(number_of_slices):
        slice_path = output_dir / f"slice_{slice_number:03d}.png"
        with slice_path.open("rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        task.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            }
        )

    return task


def generate_predictions(records, prediction_paths=PREDICTION_PATHS):
    try:
        from .slices_analyse import convert_to_png
    except ImportError:
        from slices_analyse import convert_to_png

    prediction_paths = {
        response_number: Path(path)
        for response_number, path in prediction_paths.items()
    }
    completed_by_response = {
        response_number: len(load_existing_predictions(path, records))
        for response_number, path in prediction_paths.items()
    }

    if all(
        completed == len(records)
        for completed in completed_by_response.values()
    ):
        print(
            f"Les {RESPONSE_COUNT} réponses ont déjà été générées pour les "
            f"{len(records)} patients."
        )
        return

    for response_number, completed in completed_by_response.items():
        print(
            f"Réponse {response_number}/{RESPONSE_COUNT}: "
            f"{completed}/{len(records)} patients déjà analysés."
        )

    client = create_qwen_client(timeout=300.0)

    for prediction_path in prediction_paths.values():
        prediction_path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        output_files = {
            response_number: stack.enter_context(
                prediction_paths[response_number].open("a", encoding="utf-8")
            )
            for response_number, completed in completed_by_response.items()
            if completed < len(records)
        }

        for patient_index, record in enumerate(records):
            pending_responses = [
                response_number
                for response_number, completed in completed_by_response.items()
                if patient_index >= completed
            ]
            if not pending_responses:
                continue

            patient = record["case"]
            patient_id = patient["patient_id"]
            sequence = case_sequence(patient)
            print(
                f"[{patient_index + 1}/{len(records)} - {sequence}] Conversion "
                f"des coupes du patient {patient_id}...",
                flush=True,
            )
            image_path = PROJECT_DIR / patient["image"]
            output_dir, number_of_slices = convert_to_png(
                "axial slice",
                image_path,
                root_dir=PROJECT_DIR,
            )
            task = build_model_task(output_dir, number_of_slices)

            for response_number in pending_responses:
                print(
                    f"[{patient_index + 1}/{len(records)} - {sequence}] "
                    f"Discussion indépendante {response_number}/"
                    f"{RESPONSE_COUNT}: envoi de {number_of_slices} coupes...",
                    flush=True,
                )
                response = client.chat.completions.create(
                    model=QWEN_MODEL,
                    # Une nouvelle liste de messages garantit que cette requête
                    # ne contient aucune réponse ou discussion précédente.
                    messages=[{"role": "user", "content": task}],
                )

                answer = json.loads(response.choices[0].message.content)
                prediction = normalize_prediction(answer, patient_id)
                output_file = output_files[response_number]
                output_file.write(
                    json.dumps(prediction, ensure_ascii=False) + "\n"
                )
                output_file.flush()
                print(
                    f"Réponse enregistrée dans "
                    f"{prediction_paths[response_number]}.",
                    flush=True,
                )


def empty_sequence_confusion_matrix():
    return {
        actual_sequence: {
            "T1": 0,
            "T2": 0,
            "invalide": 0,
        }
        for actual_sequence in SEQUENCES
    }


def build_sequence_confusion_matrix(records, predictions):
    matrix = empty_sequence_confusion_matrix()

    for record, prediction in zip(records, predictions):
        actual_sequence = str(record["reference"]["vertebra"]).upper()
        predicted_sequence = str(prediction.get("sequence", "")).upper()
        if predicted_sequence not in SEQUENCES:
            predicted_sequence = "invalide"

        matrix[actual_sequence][predicted_sequence] += 1

    return matrix


def merge_sequence_confusion_matrices(matrices):
    merged = empty_sequence_confusion_matrix()

    for matrix in matrices:
        for actual_sequence in SEQUENCES:
            for predicted_sequence in (*SEQUENCES, "invalide"):
                merged[actual_sequence][predicted_sequence] += matrix[
                    actual_sequence
                ][predicted_sequence]

    return merged


def sequence_accuracy(matrix):
    total = sum(sum(row.values()) for row in matrix.values())
    if total == 0:
        raise ValueError("La matrice de confusion est vide.")

    correct = sum(matrix[sequence][sequence] for sequence in SEQUENCES)
    return round(correct / total * 100, 2)


def calculate_organ_score(records, predictions):
    if len(records) != len(predictions):
        raise ValueError(
            f"{len(records)} prédictions sont requises, mais "
            f"{len(predictions)} seulement sont disponibles."
        )
    if not records:
        raise ValueError("Aucun patient à évaluer.")

    organ_scores = []
    for record, prediction in zip(records, predictions):
        expected_patient_id = str(record["case"]["patient_id"])
        if str(prediction.get("patient_id", "")) != expected_patient_id:
            raise ValueError(
                f"La prédiction ne correspond pas au patient "
                f"{expected_patient_id}."
            )

        true = record["reference"]
        true_organs = set(true["organs"])
        predicted_organs = set(prediction.get("organs", []))
        organ_denominator = len(true_organs) + len(predicted_organs)
        organ_scores.append(
            2 * len(true_organs & predicted_organs) / organ_denominator
            if organ_denominator
            else 1.0
        )

    organ_score = sum(organ_scores) / len(organ_scores)
    return round(organ_score * 100, 2)


def calculate_response_scores(case_groups, predictions):
    records = ordered_case_records(case_groups)
    if len(records) != len(predictions):
        raise ValueError(
            f"{len(records)} prédictions sont requises, mais "
            f"{len(predictions)} seulement sont disponibles."
        )

    organ_scores = {}
    offset = 0

    for sequence in SEQUENCES:
        sequence_records = case_groups[sequence]
        end = offset + len(sequence_records)
        organ_scores[sequence] = calculate_organ_score(
            sequence_records,
            predictions[offset:end],
        )
        offset = end

    sequence_matrix = build_sequence_confusion_matrix(records, predictions)

    return {
        "patients_evalues": len(records),
        "score_organes_par_sequence (en %)": organ_scores,
        "matrice_confusion_sequence": sequence_matrix,
        "score_sequence_IRM (en %)": sequence_accuracy(sequence_matrix),
    }


def calculate_combined_organ_scores(case_groups, prediction_sets):
    records = ordered_case_records(case_groups)
    combined_records = []
    combined_predictions = []

    for predictions in prediction_sets:
        if len(predictions) != len(records):
            raise ValueError(
                f"{len(records)} prédictions sont requises par réponse, mais "
                f"{len(predictions)} seulement sont disponibles."
            )
        combined_records.extend(records)
        combined_predictions.extend(predictions)

    organ_scores = {}
    offset = 0
    for sequence in SEQUENCES:
        sequence_records = case_groups[sequence]
        end = offset + len(sequence_records)
        sequence_predictions = [
            prediction
            for predictions in prediction_sets
            for prediction in predictions[offset:end]
        ]
        organ_scores[sequence] = calculate_organ_score(
            sequence_records * len(prediction_sets),
            sequence_predictions,
        )
        offset = end

    return {
        "score_organes_par_sequence (en %)": organ_scores,
        "score_organes_global (en %)": calculate_organ_score(
            combined_records,
            combined_predictions,
        ),
    }


def calculate_stability_scores(prediction_sets):
    if len(prediction_sets) < 2:
        raise ValueError(
            "Au moins deux séries de prédictions sont requises pour calculer "
            "la stabilité."
        )

    prediction_count = len(prediction_sets[0])
    if prediction_count == 0:
        raise ValueError("Aucune prédiction à comparer.")
    if any(
        len(predictions) != prediction_count
        for predictions in prediction_sets
    ):
        raise ValueError(
            "Toutes les séries doivent contenir le même nombre de prédictions."
        )

    matching_sequences = 0
    matching_organs = 0
    matching_complete_answers = 0
    compared_pairs = 0

    for prediction_index in range(prediction_count):
        patient_predictions = [
            predictions[prediction_index]
            for predictions in prediction_sets
        ]
        patient_ids = {
            str(prediction.get("patient_id", ""))
            for prediction in patient_predictions
        }
        if len(patient_ids) != 1:
            raise ValueError(
                "Les séries de prédictions ne sont pas alignées sur les "
                f"mêmes patients à la position {prediction_index + 1}."
            )

        for first_index in range(len(patient_predictions) - 1):
            first = patient_predictions[first_index]
            first_sequence = str(first.get("sequence", "")).strip().upper()
            first_organs = frozenset(first.get("organs", []))

            for second in patient_predictions[first_index + 1:]:
                second_sequence = str(
                    second.get("sequence", "")
                ).strip().upper()
                second_organs = frozenset(second.get("organs", []))

                sequence_matches = first_sequence == second_sequence
                organs_match = first_organs == second_organs
                matching_sequences += sequence_matches
                matching_organs += organs_match
                matching_complete_answers += sequence_matches and organs_match
                compared_pairs += 1

    def percentage(match_count):
        return round(match_count / compared_pairs * 100, 2)

    return {
        "paires_de_reponses_comparees": compared_pairs,
        "score_stabilite_sequence_IRM (en %)": percentage(
            matching_sequences
        ),
        "score_stabilite_organes (en %)": percentage(matching_organs),
        "score_stabilite_reponse_complete (en %)": percentage(
            matching_complete_answers
        ),
    }


def calculate_all_scores(case_groups, prediction_paths=PREDICTION_PATHS):
    records = ordered_case_records(case_groups)
    scores = {}
    sequence_matrices = []
    prediction_sets = []

    for response_number, prediction_path in prediction_paths.items():
        prediction_path = Path(prediction_path)
        if not prediction_path.exists():
            raise ValueError(
                f"Le fichier {prediction_path} est introuvable. Lancez le "
                "script sans --score-only pour générer les prédictions."
            )

        predictions = load_existing_predictions(prediction_path, records)
        response_scores = calculate_response_scores(
            case_groups,
            predictions,
        )
        scores[f"reponse_{response_number}"] = response_scores
        sequence_matrices.append(
            response_scores["matrice_confusion_sequence"]
        )
        prediction_sets.append(predictions)

    merged_matrix = merge_sequence_confusion_matrices(sequence_matrices)
    scores["synthese_3_reponses"] = {
        "predictions_evaluees": sum(
            sum(row.values()) for row in merged_matrix.values()
        ),
        **calculate_combined_organ_scores(case_groups, prediction_sets),
        "matrice_confusion_sequence": merged_matrix,
        "score_sequence_IRM (en %)": sequence_accuracy(merged_matrix),
        "stabilite_entre_reponses": calculate_stability_scores(
            prediction_sets
        ),
    }

    return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Évalue Qwen sur deux groupes équilibrés de patients CHAOS MRI."
        )
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Télécharger le dataset complet au lieu des images sélectionnées.",
    )
    parser.add_argument(
        "--patients-per-sequence",
        "--max-cases",
        dest="patients_per_sequence",
        type=int,
        default=DEFAULT_PATIENTS_PER_SEQUENCE,
        help=(
            "Nombre de patients distincts à analyser pour chacune des "
            f"séquences T1 et T2 (défaut : {DEFAULT_PATIENTS_PER_SEQUENCE})."
        ),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Calculer les scores existants sans téléchargement ni appel à Qwen.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    case_groups = load_balanced_case_groups(args.patients_per_sequence)

    if not args.score_only:
        download_dataset(case_groups, download_all=args.download_all)
        generate_predictions(ordered_case_records(case_groups))

    scores = calculate_all_scores(case_groups)
    print(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
