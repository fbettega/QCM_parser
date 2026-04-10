import cv2
import numpy as np
from pdf2image import convert_from_path

# 1. Charger le PDF
pages = convert_from_path("./data/DOC070426-07042026153402.pdf", dpi=300)
img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

# 2. Dessiner des repères visuels pour tester tes coordonnées
# On va dessiner un rectangle là où tu penses que se trouve l'identité
# ZONE_IDENTITE = (600, 200, 1800, 400) -> (x1, y1, x2, y2)
cv2.rectangle(img, (600, 200), (1800, 400), (0, 255, 0), 3)

# On dessine aussi une grille de repère (tous les 100px)
for i in range(0, img.shape[1], 100):
    cv2.line(img, (i, 0), (i, img.shape[0]), (200, 200, 200), 1)
for i in range(0, img.shape[0], 100):
    cv2.line(img, (0, i), (img.shape[1], i), (200, 200, 200), 1)

# 3. SAUVEGARDER au lieu d'afficher
cv2.imwrite("data/test_calibration.jpg", img)

print("Image 'test_calibration.jpg' générée. Ouvre-la pour vérifier l'encadré vert.")