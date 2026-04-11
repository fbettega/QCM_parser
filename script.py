import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import re
import os

# --- CONFIGURATION (À ADAPTER SELON VOS SCANS) ---
# Chemin vers l'exécutable Tesseract (si sur Windows, décommentez et adaptez)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Seuil de noirceur pour considérer une case comme cochée (entre 0 et 1)
SEUIL_REMPLISSAGE = 0.4 

# Coordonnées estimées (X, Y, Largeur, Hauteur) pour l'en-tête
#ZONE_IDENTITE = (600, 200, 1800, 400)
# 81 et 115 sur 1224

ZONE_IDENTITE = (0.06, 0.10)

def traiter_pdf(chemin_pdf):
    """Convertit le PDF en images."""
    return convert_from_path(chemin_pdf, dpi=300)

def redresser_image(image_cv):
    """Détecte l'inclinaison de l'image et la redresse."""
    # 1. Conversion en gris et détection de bords
    gris = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    bords = cv2.Canny(gris, 50, 150, apertureSize=3)

    # 2. Détection des lignes avec la transformée de Hough
    # On cherche des lignes assez longues (> 100px)
    lignes = cv2.HoughLinesP(bords, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    angles = []
    if lignes is not None:
        for ligne in lignes:
            x1, y1, x2, y2 = ligne[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # On ne garde que les lignes quasi-horizontales (entre -10 et 10 deg)
            # ou quasi-verticales (on les remet sur l'axe horizontal pour la moyenne)
            if -10 < angle < 10:
                angles.append(angle)
            elif 80 < angle < 100:
                angles.append(angle - 90)
            elif -100 < angle < -80:
                angles.append(angle + 90)

    # 3. Calcul de l'angle médian (plus robuste que la moyenne)
    if len(angles) > 0:
        angle_final = np.median(angles)
    else:
        angle_final = 0

    # 4. Rotation de l'image pour la remettre droite
    (h, w) = image_cv.shape[:2]
    centre = (w // 2, h // 2)
    matrice = cv2.getRotationMatrix2D(centre, angle_final, 1.0)
    image_rectifiee = cv2.warpAffine(image_cv, matrice, (w, h), 
                                     flags=cv2.INTER_CUBIC, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(255, 255, 255)) # Fond blanc

    return image_rectifiee, angle_final


def est_cochee(img_binaire, x_centre, y_centre, w, h):
    """Calcule le ratio de remplissage d'une case après nettoyage."""
    x1 = int(x_centre - w // 2)
    y1 = int(y_centre - h // 2)
    x2 = x1 + w
    y2 = y1 + h

    roi = img_binaire[max(0, y1):y2, max(0, x1):x2]
    
    if roi.size == 0:
        return 0.0

    # --- NETTOYAGE DU BRUIT ---
    # Supprime les pixels isolés (résidus des lettres rouges ou poussière)
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

    total_pixels = roi.shape[0] * roi.shape[1]
    pixels_blancs = cv2.countNonZero(roi)
    
    return pixels_blancs / total_pixels

def extraire_texte(image_couleur, h, w,  rel_y1, rel_y2):
    """Extrait le texte et sépare intelligemment chiffres et lettres."""
    
    # 1. Découpage du bloc identité
    y1, y2 = int(rel_y1 * h), int(rel_y2 * h)
    roi = image_couleur[y1:y2, :]

    if roi.size == 0:
        return "N/A", "N/A"

    # 2. Prétraitement pour booster l'OCR
    # On passe en gris
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # CRUCIAL : On agrandit l'image (x2 ou x3) pour que Tesseract "voit" mieux les caractères
    gris = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Binarisation adaptative (mieux que le seuil fixe à 150 pour les scans variables)
    binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 3. Extraction globale (Lettres + Chiffres)
    # --psm 7 est souvent meilleur pour une seule ligne de texte
    configuration = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    texte_brut = pytesseract.image_to_string(binaire, config=configuration).upper()

    # 4. Tri par Expressions Régulières (Regex)
    
    # On cherche le numéro d'étudiant : un bloc de chiffres (souvent 7 ou 8 chiffres)
    # On prend tous les chiffres collés ou non
    tous_les_chiffres = "".join(re.findall(r'\d', texte_brut))
    num_etudiant = tous_les_chiffres if tous_les_chiffres else "Inconnu"

    # On cherche le Nom/Prénom : tout ce qui n'est pas un chiffre, nettoyé
    # On remplace les chiffres par rien pour ne garder que le texte
    nom_prenom = re.sub(r'[^A-Z\s]', '', texte_brut)
    # On nettoie les espaces multiples
    nom_prenom = " ".join(nom_prenom.split()).strip()
    
    if not nom_prenom:
        nom_prenom = "Inconnu"

    return num_etudiant, nom_prenom


def detecter_lignes_y(binaire_omr, w,nom_prenom):
    
    # 1. Analyse de la bande de synchronisation à droite
    bande_start = int(w * 0.961) 
    band_end = int(w * 0.986)

    bande_droite = binaire_omr[:, bande_start:band_end]
    
    # Projection horizontale : on normalise entre 0 et 1
    # On suppose que 255 (blanc) est la couleur du tick
    projection_noir = np.mean(bande_droite, axis=1) / 255
    
    # 2. Seuillage
    # On obtient un tableau de booléens (True si c'est un pixel de tick)
    seuillage_ticks = projection_noir > 0.5 
    
    tous_les_ticks = []
    y_debut = None
    count = 0
    
    # 3. Identification des blocs (début et fin)
    for y, is_tick in enumerate(seuillage_ticks):
        if is_tick:
            if count == 0:
                y_debut = y  # On enregistre le premier pixel du bloc
            count += 1
        elif count > 0:
            # On vient de sortir d'un bloc noir
            y_fin = y - 1  # La fin est le pixel juste avant le changement
            
            # FILTRE DE TAILLE : on vérifie la hauteur du bloc trouvé
            if 10 < count < 100:
                # On ajoute un tuple (début, fin)
                tous_les_ticks.append((y_debut, y_fin))
            
            # Réinitialisation pour le prochain bloc
            count = 0
            y_debut = None
    # --- GESTION DU RETOUR ---
    # Structure attendue : 43 marques
    if len(tous_les_ticks) >= 43:
        return tous_les_ticks[:43]
    else:
        print(f"Attention {nom_prenom}: {len(tous_les_ticks)} marques trouvées au lieu de 43.")
        return tous_les_ticks

def detecter_colonnes_x(binaire_omr, w,paires_y,nom_prenom):
    # 1. On définit la zone verticale globale des questions
    # On ignore les 2 premiers ticks (Start) et le dernier (Final) pour n'avoir que les questions
    y_start = paires_y[2][0] 
    y_end = paires_y[-1][0]
    
    zone_questions = binaire_omr[y_start:y_end,:int(w * 0.961) ]
    h_zone, w_zone = zone_questions.shape
    # if nom_prenom == "SOUMARE NIOUMA":
    #     print(f"DEBUG {nom_prenom}: zone_questions shape = {zone_questions.shape}")

    # 2. Projection verticale : on fait la moyenne sur la hauteur
    # Comme l'encre est blanche (255), les colonnes de cases créeront des "pics"
    kernel = np.ones((1, int(w * 0.02)), np.uint8)
    zone_fusionnee = cv2.dilate(zone_questions, kernel, iterations=1)
    projection = np.mean(zone_fusionnee, axis=0) / 255
    
    # 3. Seuillage pour identifier les zones de texte/cases
    # Un seuil de 0.02 (2% de pixels blancs sur la colonne) suffit souvent
    seuil = 0.02
    colonnes_actives = projection > seuil
    
    paires_x = []
    x_debut = None
    en_colonne = False

    for x, actif in enumerate(colonnes_actives):
            if actif and not en_colonne:
                x_debut = x
                en_colonne = True
            elif not actif and en_colonne:
                largeur = x - x_debut
                if largeur > 50: # On filtre les petits bruits
                    paires_x.append((x_debut, x))
                en_colonne = False
                
        # --- LA CORRECTION MAGIQUE ---
        # Si à la fin de l'image on est encore "en colonne", on la ferme manuellement !
    if en_colonne:
        paires_x.append((x_debut, len(colonnes_actives)))
        # Sécurité : on attend 5 colonnes
    if len(paires_x) != 5:
        print(f"Attention {nom_prenom}: {len(paires_x)} colonnes trouvées. Vérifiez le seuil.")
        
    return paires_x 

def analyser_feuille(image_cv_originale):
    # --- 0. REDRESSEMENT ---
    image_cv, angle = redresser_image(image_cv_originale)
    if abs(angle) > 0.05:
        print(f"Correction de l'inclinaison : {angle:.2f}°")
    
    h, w = image_cv.shape[:2]
    donnees_feuille = []
    
    # --- 1. PRÉTRAITEMENT LECTURE (OMR) ---
    # Utilisation du canal ROUGE (index 2) pour effacer la grille rouge
    canal_rouge = image_cv[:, :, 2]
    flou = cv2.GaussianBlur(canal_rouge, (5, 5), 0)
    # On binarise pour isoler le stylo sombre
    _, binaire_omr = cv2.threshold(flou, 195, 255, cv2.THRESH_BINARY_INV)

    # --- 2. PRÉTRAITEMENT STRUCTURE (COLONNES) ---
    canal_vert = image_cv[:, :, 1]
    flou_v = cv2.GaussianBlur(canal_vert, (5, 5), 0)
    _, binaire_struct = cv2.threshold(flou_v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extraction en-tête et géométrie
    num_etudiant, nom_prenom = extraire_texte(image_cv, h, w, *ZONE_IDENTITE)
    paires_y = detecter_lignes_y(binaire_omr, w, nom_prenom)
    paires_x = detecter_colonnes_x(binaire_struct, w, paires_y, nom_prenom)

    # Paramètres de mesure
    RATIO_REPENTIR = 0.151
    RATIO_A = 0.302
    RATIO_PAS = 0.151
    TAILLE_REL = 0.12

    for bloc_idx, (x_start, x_end) in enumerate(paires_x):
        largeur_col = x_end - x_start
        tw = th = int(largeur_col * TAILLE_REL)

        for q_in_col in range(20):
            q_numero = (bloc_idx * 20) + (q_in_col + 1)
            idx_l1 = 2 + (q_in_col * 2)
            idx_l2 = idx_l1 + 1
            if idx_l2 >= len(paires_y): break

            y_l1 = (paires_y[idx_l1][0] + paires_y[idx_l1][1]) // 2
            y_l2 = (paires_y[idx_l2][0] + paires_y[idx_l2][1]) // 2

            centres_x_abcde = [int(x_start + (largeur_col * (RATIO_A + i * RATIO_PAS))) for i in range(5)]
            x_rep = int(x_start + (largeur_col * RATIO_REPENTIR))

# --- ANALYSE LIGNE 1 ---
            ratios_l1 = [est_cochee(binaire_omr, cx, y_l1, tw, th) for cx in centres_x_abcde]
            
            # 1. On prend la valeur la plus basse de la ligne
            valeur_min_l1 = min(ratios_l1)
            # 2. On la plafonne à 3% (0.03) max pour éviter que 5 cases cochées ne faussent le calcul
            bruit_de_fond_l1 = min(valeur_min_l1, 0.03)
            
            # Une case est cochée si elle a au moins 4% ET qu'elle dépasse le (bruit + 5%)
            res_l1 = [ (r > 0.04 and r > (bruit_de_fond_l1 + 0.05)) for r in ratios_l1 ]

            if any(res_l1):
                donnees_feuille.append({
                    'numéro étudiants': num_etudiant, 'nom': nom_prenom, 'question': q_numero,
                    'ligne réponse': 1, 'repentance': False,
                    'Réponse A': res_l1[0], 'Réponse B': res_l1[1], 'Réponse C': res_l1[2], 
                    'Réponse D': res_l1[3], 'Réponse E': res_l1[4]
                })

            # --- ANALYSE LIGNE 2 (REPENTIR) ---
            ratio_rep = est_cochee(binaire_omr, x_rep, y_l2, tw, th)
            ratios_l2 = [est_cochee(binaire_omr, cx, y_l2, tw, th) for cx in centres_x_abcde]
            
            valeur_min_l2 = min(ratios_l2)
            bruit_de_fond_l2 = min(valeur_min_l2, 0.03)
            
            is_rep_coche = (ratio_rep > 0.08)
            res_l2 = [ (r > 0.04 and r > (bruit_de_fond_l2 + 0.05)) for r in ratios_l2 ]

            if is_rep_coche or any(res_l2):
                donnees_feuille.append({
                    'numéro étudiants': num_etudiant, 'nom': nom_prenom, 'question': q_numero,
                    'ligne réponse': 2, 'repentance': is_rep_coche,
                    'Réponse A': res_l2[0], 'Réponse B': res_l2[1], 'Réponse C': res_l2[2], 
                    'Réponse D': res_l2[3], 'Réponse E': res_l2[4]
                })

    return donnees_feuille

def process_tous_les_pdfs(dossier_pdfs, fichier_sortie_csv):
    """Parcourt tous les PDF, les analyse et exporte le CSV."""
    toutes_les_donnees = []
    
    for fichier in os.listdir(dossier_pdfs):
        if fichier.lower().endswith('.pdf'):
            chemin_complet = os.path.join(dossier_pdfs, fichier)
            print(f"Traitement de {fichier}...")
            images = traiter_pdf(chemin_complet)
            
            for img in images:
                # Convertir l'image PIL en image OpenCV
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                donnees = analyser_feuille(img_cv)
                toutes_les_donnees.extend(donnees)
                
    # Création du CSV avec Pandas
    df = pd.DataFrame(toutes_les_donnees)
    df.to_csv(fichier_sortie_csv, index=False, encoding='utf-8-sig', sep=';')
    print(f"Terminé ! Sauvegardé dans {fichier_sortie_csv}")

# --- EXÉCUTION ---
process_tous_les_pdfs('./data', 'outpout/resultats_examen.csv')