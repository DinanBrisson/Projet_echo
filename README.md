# Projet_echo

## Description du Projet
Ce projet vise à segmenter des images échographiques en utilisant des architectures de Deep Learning basées sur U-Net. 
Deux versions du modèle sont explorées :
- Méthode A : U-Net avec ResNet-50 comme encodeur
- Méthode B : U-Net avec VGG16 comme encodeur
L'objectif est d'améliorer la détection automatique des structures échographiques en comparant ces deux méthodes.

## Installation et Configuration
git clone https://github.com/DinanBrisson/Projet_echo.git

cd Projet_echo

pip install -r requirements.txt

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


