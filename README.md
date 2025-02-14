# Projet_echo

## Description du Projet
Ce projet vise à segmenter des images échographiques en utilisant des architectures de Deep Learning basées sur U-Net. 
Deux versions du modèle sont explorées :
- Méthode A : U-Net avec ResNet-50 comme encodeur
- Méthode B : U-Net avec VGG16 comme encodeur
L'objectif est d'améliorer la détection automatique des structures échographiques en comparant ces deux méthodes.

## Organisation des Fichiers
📂 Projet_echo/
│-- 📂 app/                    # Contient l'application principale (ex: Streamlit)
│-- 📂 notebooks/               # Notebooks Jupyter pour l'entraînement
│-- 📂 scripts/                 # Scripts principaux du projet
│   ├── 📂 Data/                # Contient les scripts de gestion des données
│   │   ├── image_processor.py# Pré-traitement des images
│       ├── unet.py               # Script d'entraînement et d'évaluation
│   ├── script.py                 # Implémentation du modèle U-Net
│-- 📄 .gitignore               # Fichiers à ignorer par Git
│-- 📂 models/                  # Lien drive pour télécharger les modèles entrainés
│-- 📂 data/                  # Lien drive pour télécharger les données originales et train/test/val
│-- 📄 requirements.txt         # Liste des dépendances


## Installation et Configuration
git clone https://github.com/DinanBrisson/Projet_echo.git
cd Projet_echo

Télécharger les données originales :
Les images originales doivent être téléchargés depuis le lien suivant et placés dans le dossier data/

Télécharger les modèles pré-entraînés :
Les modèles nécessaires doivent être téléchargés depuis le lien suivant et placés dans le dossier models/

## Éxécution
Lancement du script (processing des données et entrainement/évaluation du modèle choisi)
python scripts/script.py

Lacement de l'app : 
python app/app.py

## Licence

Dinan BRISSON - M2 TMS ISEN


