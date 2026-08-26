# LLM_project

Outil Python d'inspection de datasets d'imagerie médicale et d'évaluation d'un
modèle vision-langage. Le projet regroupe deux workflows complémentaires :

1. analyser la structure et la qualité de datasets CT au format NIfTI ou
   DICOM, puis générer un rapport HTML/PDF ;
2. évaluer Qwen sur le dataset CHAOS MRI pour la reconnaissance des organes et
   des séquences T1/T2.

## Fonctionnalités

### Analyse de datasets médicaux

- détection des dossiers d'entraînement, de test et de validation ;
- prise en charge des structures standards `imagesTr`, `labelsTr` et
  `imagesTs`, ainsi que de noms de dossiers non standards ;
- association automatique des images et des masques ;
- détection probable du type de tâche : segmentation, classification ou
  inconnu ;
- lecture des volumes `.nii` et `.nii.gz` avec NiBabel ;
- conversion automatique des séries DICOM CT valides vers NIfTI ;
- extraction des dimensions, du nombre de coupes, de l'espacement des voxels,
  de l'épaisseur des coupes, de l'orientation, du type numérique, de la matrice
  affine, de la taille physique et des besoins mémoire ;
- calcul de statistiques d'intensité et vérification des valeurs non finies ;
- contrôle de l'alignement image/masque et de la cohérence entre patients ;
- génération de rapports HTML et, en option, PDF ;
- interface graphique pour sélectionner un dataset, consulter le rapport et
  envoyer des coupes à Qwen.

### Évaluation CHAOS MRI

- sélection équilibrée de patients T1 et T2 ;
- trois requêtes indépendantes par patient ;
- comparaison des organes détectés avec `dice.jsonl` ;
- matrice de confusion T1/T2 pour chaque réponse et pour la synthèse ;
- mesure de la stabilité entre les trois réponses.

## Arborescence

```text
LLM_project/
├── src/
│   ├── check_structure_dataset.py
│   ├── dataset_analyzer.py
│   ├── general_conversion.py
│   ├── interface.py
│   ├── nifti_analyzer.py
│   ├── report.py
│   └── test_accuracy_evaluation/
│       ├── api_config.py
│       ├── slices_analyse.py
│       ├── true_data.py
│       ├── train.jsonl
│       ├── dice.jsonl
│       └── predictions_1.jsonl ... predictions_3.jsonl
├── data/
│   ├── dataset_analysis/
│   ├── evaluation/
│   ├── interface/
│   ├── nii/
│   └── slices/
├── dataset/                 # datasets médicaux locaux
├── docs/                    # documentation détaillée et rapports
├── examples/                # démonstrations API, NIfTI et PDF
└── requirements/            # dépendances séparées par workflow
```

Les datasets locaux, les clés API et les coupes PNG générées sont ignorés par
Git.

## Prérequis

- Python 3.10 ou plus récent ;
- Tkinter pour l'interface graphique ;
- suffisamment d'espace disque pour les datasets médicaux ;
- une clé API Qwen uniquement pour les fonctionnalités VLM.

Sous Ubuntu, installez le paquet Tkinter correspondant à votre version de
Python, par exemple :

```bash
sudo apt install python3-tk
```

## Installation

Créez et activez un environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sous Windows :

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Pour installer l'environnement complet :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Des groupes plus légers sont également disponibles :

| Besoin | Commande |
|---|---|
| Analyse NIfTI/DICOM | `python -m pip install -r requirements/core.txt` |
| Interface graphique | `python -m pip install -r requirements/interface.txt` |
| Évaluation Qwen | `python -m pip install -r requirements/interface.txt -r requirements/evaluation.txt` |
| Modèle local optionnel | `python -m pip install -r requirements/local-model.txt` |

## Configuration de Qwen

Cette étape est nécessaire uniquement pour l'analyse VLM et l'évaluation
CHAOS :

```bash
cp .env.example .env
```

Modifiez ensuite `.env` :

```dotenv
QWEN_API_KEY=replace-with-your-api-key
QWEN_BASE_URL=https://example.com/v1
QWEN_MODEL=qwen-model-name
```

Le fichier `.env` est ignoré par Git. Ne publiez jamais une clé réelle.

## Formats supportés

L'analyse accepte directement :

- `.nii` ;
- `.nii.gz` ;
- des séries DICOM `.dcm` contenant suffisamment de coupes CT cohérentes.

Les séries DICOM sont converties dans un dossier `converted_nifti/` avant
l'analyse. Les localizers, scouts et séries trop courtes sont ignorés. En cas
d'échec bloquant, un diagnostic est écrit dans :

```text
data/dataset_analysis/Warning.txt
```

## Utilisation de l'analyse de dataset

Exécutez les commandes depuis la racine du dépôt.

### Interface graphique

```bash
python -m src.interface
```

L'interface permet de sélectionner un dataset, choisir un split, lancer
l'analyse, consulter le rapport HTML et explorer les coupes NIfTI.

### 1. Vérifier la structure

Cette commande inspecte les dossiers et les associations image/masque sans lire
les valeurs des voxels :

```bash
python -m src.check_structure_dataset dataset/Task02_Heart
```

Pour afficher la structure complète en JSON :

```bash
python -m src.check_structure_dataset dataset/Task02_Heart --json
```

### 2. Analyser un patient

Sans masque :

```bash
python -m src.nifti_analyzer \
  dataset/Task02_Heart/imagesTr/la_003.nii.gz
```

Avec son masque :

```bash
python -m src.nifti_analyzer \
  dataset/Task02_Heart/imagesTr/la_003.nii.gz \
  --label dataset/Task02_Heart/labelsTr/la_003.nii.gz
```

La sortie par défaut est :

```text
data/dataset_analysis/CT_data.json
```

Un autre chemin peut être choisi avec `--output`.

### 3. Analyser un dataset complet

Le split `train` est utilisé par défaut :

```bash
python -m src.dataset_analyzer dataset/Task02_Heart
```

Choisir explicitement un split :

```bash
python -m src.dataset_analyzer dataset/Task02_Heart --split test
```

Analyser tous les splits détectés :

```bash
python -m src.dataset_analyzer dataset/Task02_Heart --split all
```

Afficher le résultat complet dans le terminal :

```bash
python -m src.dataset_analyzer dataset/Task02_Heart --json
```

La sortie JSON par défaut est :

```text
data/dataset_analysis/analyse_dataset.json
```

### 4. Générer le rapport

À partir du JSON par défaut :

```bash
python -m src.report
```

Avec un rapport PDF supplémentaire :

```bash
python -m src.report --pdf data/dataset_analysis/report.pdf
```

Le rapport présente notamment la couverture des annotations, les données
manquantes, l'alignement image/masque, les dimensions, l'orientation,
l'espacement des voxels, les statistiques d'intensité, les avertissements et
les recommandations de prétraitement.

## Évaluation de Qwen sur CHAOS MRI

Les références et prédictions se trouvent dans :

```text
src/test_accuracy_evaluation/
```

Le rôle des fichiers JSONL est le suivant :

- `train.jsonl` : chemins des volumes, patient, modalité et annotations ;
- `dice.jsonl` : vérité terrain des organes et de la séquence T1/T2 ;
- `predictions_1.jsonl` à `predictions_3.jsonl` : réponses indépendantes du
  modèle.

### Lancer une évaluation

Par défaut, cinq patients T1 et cinq autres patients T2 sont envoyés trois fois
au modèle. Seules les images nécessaires sont téléchargées :

```bash
python -m src.test_accuracy_evaluation.true_data
```

Changer le nombre de patients par séquence :

```bash
python -m src.test_accuracy_evaluation.true_data \
  --patients-per-sequence 8
```

Recalculer uniquement les scores existants, sans téléchargement ni appel API :

```bash
python -m src.test_accuracy_evaluation.true_data --score-only
```

Les scores et les matrices de confusion sont affichés en JSON dans le terminal.

Télécharger le dataset CHAOS MRI complet, environ 102 MiB :

```bash
python -m src.test_accuracy_evaluation.true_data --download-all
```

### Scores calculés

- Le score des organes est un coefficient de Sørensen-Dice appliqué aux
  ensembles de noms d'organes. Ce n'est pas un Dice spatial de segmentation.
- Le score de séquence est le pourcentage de prédictions situées sur la
  diagonale T1/T2 de la matrice de confusion.
- Une prédiction de séquence différente de `T1` ou `T2` est placée dans la
  colonne `invalide`.
- La stabilité mesure l'accord entre toutes les paires formées par les trois
  réponses d'un même patient.

Les références actuelles contiennent les quatre mêmes organes pour chaque
volume. Le score des organes doit donc être interprété avec prudence : un modèle
répondant toujours avec ces quatre organes peut obtenir 100 % sans démontrer une
véritable capacité de localisation.

## Résultats générés

| Fichier ou dossier | Contenu |
|---|---|
| `data/dataset_analysis/CT_data.json` | Analyse d'un patient |
| `data/dataset_analysis/analyse_dataset.json` | Analyse d'un dataset complet |
| `data/dataset_analysis/report.html` | Rapport HTML du dataset |
| `data/dataset_analysis/report.pdf` | Rapport PDF optionnel |
| `data/slices/` | Coupes PNG temporaires envoyées au VLM |
| `data/nii/CHAOS/` | Volumes CHAOS téléchargés localement |

## Confidentialité et sécurité

- Les clés sont chargées depuis `.env` et ne doivent jamais être commitées.
- Les datasets locaux sont ignorés par Git.
- Les commandes d'analyse structurelle et NIfTI sont locales.
- Les fonctions Qwen envoient les coupes sélectionnées au serveur configuré
  par `QWEN_BASE_URL`. Vérifiez les règles de confidentialité applicables avant
  d'envoyer des données médicales.

## Documentation complémentaire

- [Guide détaillé de l'analyse de dataset](docs/dataset_analysis.md)
- [Organisation des dépendances](requirements/README.md)
- [Organisation des données](data/README.md)
