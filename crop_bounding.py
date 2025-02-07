import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
import re

# Dossiers
data_folder = "Data/"
annotated_folder = "Data_Annoted/"
cropped_folder_gray = "Cropped_Gray/"

# Créer les dossiers de sortie s'ils n'existent pas
os.makedirs(annotated_folder, exist_ok=True)
os.makedirs(cropped_folder_gray, exist_ok=True)

# Fonction de tri naturel pour respecter l'ordre des fichiers (1_1, 1_2, 2_1, ...)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

# Fonction pour extraire les annotations du fichier XML
def extract_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}

    for mark in root.findall(".//mark"):
        image_id = mark.find("image").text.strip()  # ID de l'image annotée
        svg_data = mark.find("svg").text

        if svg_data:
            try:
                svg_data = json.loads(svg_data)  # Convertir JSON en liste Python
                points = [(point["x"], point["y"]) for point in svg_data[0]["points"]]
                if image_id in annotations:
                    annotations[image_id].append(points)
                else:
                    annotations[image_id] = [points]  # Stocker les annotations par ID d'image
            except json.JSONDecodeError:
                print(f"Erreur de parsing JSON dans {xml_path}")

    return annotations

# Lister et trier tous les fichiers XML et images de manière naturelle
xml_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".xml")], key=natural_sort_key)
image_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".jpg") or f.endswith(".png")], key=natural_sort_key)

# Associer chaque fichier XML à ses images correspondantes
xml_to_images = {}
for image_file in image_files:
    base_name = image_file.split("_")[0]  # Extrait "1" de "1_1.jpg"
    if base_name in xml_to_images:
        xml_to_images[base_name].append(image_file)
    else:
        xml_to_images[base_name] = [image_file]

# Parcourir tous les fichiers XML dans l'ordre
for xml_file in xml_files:
    xml_path = os.path.join(data_folder, xml_file)
    base_name = xml_file.replace(".xml", "")

    # Charger les annotations
    annotations = extract_annotations(xml_path)

    if not annotations:
        print(f"Aucune annotation trouvée dans {xml_file}.")
        continue

    # Vérifier si des images correspondent à ce fichier XML
    if base_name not in xml_to_images:
        print(f"Aucune image trouvée pour {xml_file}.")
        continue

    for image_file in xml_to_images[base_name]:
        image_path = os.path.join(data_folder, image_file)

        # Charger l'image en couleur
        img = cv2.imread(image_path)
        if img is None:
            print(f"Impossible de charger {image_file}.")
            continue

        # Récupérer l'ID d'image correct depuis le XML
        annotation_key = list(annotations.keys())[0]

        # **1. Enregistrer l'image originale avec annotation en couleur (non croppée)**
        img_annotated = img.copy()
        for points in annotations[annotation_key]:
            points = np.array(points, np.int32)
            cv2.polylines(img_annotated, [points], isClosed=True, color=(255, 0, 0), thickness=2)

        annotated_filename = f"{image_file.replace('.jpg', '_annotated.jpg').replace('.png', '_annotated.png')}"
        annotated_path = os.path.join(annotated_folder, annotated_filename)
        cv2.imwrite(annotated_path, img_annotated)

        print(f"Image annotée sauvegardée sans crop : {annotated_filename}")

        # **2. Appliquer le crop et enregistrer en niveaux de gris sans annotation**
        for idx, points in enumerate(annotations[annotation_key], start=1):
            points = np.array(points, np.int32)

            # Trouver la boîte englobante
            x, y, w, h = cv2.boundingRect(points)

            # Vérifier que la boîte est correcte
            if w > 0 and h > 0:
                cropped_img = img[y:y + h, x:x + w].copy()  # Crop de l'image

                # Vérifier que le crop est valide
                if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    # Redimensionner en 256x256
                    cropped_img_resized = cv2.resize(cropped_img, (256, 256))

                    # Convertir en niveaux de gris avec 1 canal
                    cropped_img_gray = cv2.cvtColor(cropped_img_resized, cv2.COLOR_BGR2GRAY)

                    # Vérification du nombre de canaux après conversion
                    if len(cropped_img_gray.shape) == 2:
                        print(f"Image {image_file} bien convertie en niveaux de gris avec 1 canal")

                    # Sauvegarde de l'image en 1 canal (niveaux de gris)
                    cropped_filename_gray = f"{image_file.replace('.jpg', '_gray.jpg').replace('.png', '_gray.png')}"
                    cropped_path_gray = os.path.join(cropped_folder_gray, cropped_filename_gray)
                    cv2.imwrite(cropped_path_gray, cropped_img_gray)

                    print(f"Image croppée sauvegardée en niveaux de gris : {cropped_filename_gray}")

                    # Affichage du crop en niveaux de gris
                    plt.figure(figsize=(4, 4))
                    plt.imshow(cropped_img_gray, cmap="gray")
                    plt.axis("off")
                    plt.title(f"{image_file} (Niveaux de gris)")
                    plt.show()
                else:
                    print(f"Image croppée invalide pour {image_file}.")
            else:
                print(f"Boîte englobante invalide pour {image_file}.")

print("Tous les fichiers ont été générés :")
print(f"- Images annotées sans crop dans '{annotated_folder}'")
print(f"- Images croppées en niveaux de gris dans '{cropped_folder_gray}'")
