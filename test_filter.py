import cv2
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
img = cv2.imread("Cropped_Gray/1_1_gray.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Appliquer un Filtre Médian plus large pour réduire le bruit speckle
img_median = cv2.medianBlur(img, 3)

# 2. Appliquer CLAHE avec des paramètres ajustés
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
img_clahe = clahe.apply(img_median)

# Affichage des résultats
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(img_median, cmap="gray")
axes[1].set_title("Filtre Médian")
axes[1].axis("off")

axes[2].imshow(img_clahe, cmap="gray")
axes[2].set_title("CLAHE (Paramètres ajustés)")
axes[2].axis("off")

plt.show()