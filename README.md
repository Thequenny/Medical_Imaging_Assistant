# LLM_project

Interface graphique pour inspecter un dataset d'imagerie médicale, générer son
rapport de qualité et discuter des coupes sélectionnées avec Qwen.

La documentation détaillée de l'analyse NIfTI/DICOM et des commandes en ligne
se trouve dans [docs/dataset_analysis.md](docs/dataset_analysis.md).

## Prérequis

- Python 3.10 ou plus récent ;
- Tkinter pour l'interface graphique.

Sous Ubuntu, installez Tkinter avec :

```bash
sudo apt install python3-tk
```

## Installation

Depuis la racine du projet, créez et activez un environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/interface.txt
```

Sous Windows :

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/interface.txt
```

## Configuration de Qwen

Cette étape est facultative pour l'analyse locale du dataset, mais nécessaire
pour envoyer des coupes au modèle depuis l'interface :

```bash
cp .env.example .env
```

Renseignez ensuite les paramètres dans `.env` :

```dotenv
QWEN_API_KEY=replace-with-your-api-key
QWEN_BASE_URL=https://example.com/v1
QWEN_MODEL=qwen-model-name
```

Le fichier `.env` est ignoré par Git. Ne publiez jamais une clé réelle.

## Lancer l'interface

Activez l'environnement virtuel, placez-vous à la racine du projet, puis
exécutez :

```bash
python -m src.interface
```

Dans l'interface :

1. choisissez le split à analyser (`train`, `test`, `validation` ou `all`) ;
2. cliquez sur **Select and analyze folder** et sélectionnez le dossier du
   dataset ;
3. ouvrez le rapport généré avec **Open generated report** ;
4. utilisez l'espace d'exploration pour afficher les coupes NIfTI et, si Qwen
   est configuré, les commenter avec le modèle.

Les rapports de l'interface sont enregistrés dans :

```text
data/dataset_analysis/
```

## Données acceptées

L'interface prend en charge les volumes `.nii`, `.nii.gz` et les séries DICOM
`.dcm`. Les détails sur la détection des datasets, la conversion DICOM et le
contenu des rapports sont disponibles dans la
[documentation d'analyse](docs/dataset_analysis.md).

## Confidentialité

L'analyse du dataset est locale. Seules les coupes explicitement envoyées à
Qwen quittent la machine et sont transmises au serveur défini par
`QWEN_BASE_URL`. Vérifiez les règles de confidentialité applicables avant
d'envoyer des données médicales.

## Documentation

- [Analyse des datasets NIfTI/DICOM](docs/dataset_analysis.md)
- [Organisation des dépendances](requirements/README.md)
- [Organisation des données](data/README.md)
