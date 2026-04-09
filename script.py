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
ZONE_IDENTITE = (600, 200, 1800, 400)






def traiter_pdf(chemin_pdf):
    """Convertit le PDF en images."""
    return convert_from_path(chemin_pdf, dpi=300)

def est_cochee(img_binaire, x, y, w, h):
    """Vérifie si une case spécifique est cochée en comptant les pixels noirs."""
    roi = img_binaire[y:y+h, x:x+w]
    total_pixels = w * h
    # Dans img_binaire (inversée), les marques noires sont blanches (255)
    pixels_blancs = cv2.countNonZero(roi)
    ratio = pixels_blancs / total_pixels
    return ratio > SEUIL_REMPLISSAGE

def extraire_texte(image_couleur, x, y, x2, y2):
    """Extrait le texte à partir de l'image BGR originale."""
    # Découpage du bloc identité
    roi = image_couleur[y:y2, x:x2]
    
    # Sécurité si le ROI est vide
    if roi.size == 0:
        return "N/A", "N/A"
    
    # Prétraitement
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binaire = cv2.threshold(gris, 150, 255, cv2.THRESH_BINARY)

    # 1. Extraction du NUMÉRO
    conf_num = "--psm 11 -c tessedit_char_whitelist=0123456789"
    raw_num = pytesseract.image_to_string(binaire, config=conf_num)
    num_etudiant = "".join(re.findall(r'\d+', raw_num))

    # 2. Extraction NOM PRÉNOM
    conf_nom = "--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    raw_nom = pytesseract.image_to_string(binaire, config=conf_nom)
    nom_prenom = " ".join(raw_nom.split()).strip()

    return num_etudiant, nom_prenom

def detecter_lignes_y(binaire_omr):
    h, w = binaire_omr.shape
    # 1. Analyse de la bande de synchronisation à droite
    # On cible la zone où se trouvent les rectangles noirs (ticks)
    # bande_droite = binaire_omr[:, w-70:w-10]
    largeur_bande = int(w * 0.05) 
    bande_droite = binaire_omr[:, w - largeur_bande:]
    # Projection horizontale : on compte les pixels noirs (0) ou blancs (255) 
    # selon si votre image est inversée. Ici on suppose 255 = blanc (le tick).
    projection_noir = np.mean(bande_droite, axis=1)/ 255
    
    # On considère que c'est un tick si plus de 50% de la largeur de la bande est noire
    seuil = projection_noir > 0.5 # bande_droite.shape[1] * 0.5
    seuillage_ticks = projection_noir > seuil
    tous_les_ticks = []
    current_y_sum, count = 0, 0
    
    # 2. Identification des centres des blocs noirs
    # 3. Parcours des lignes
    for y, is_tick in enumerate(seuillage_ticks):
        if is_tick:
            current_y_sum += y
            count += 1
        elif count > 0:
            # FILTRE DE TAILLE : Un vrai tick fait entre 3 et 25 pixels de haut
            # Cela permet d'éliminer les 2 marques parasites (poussières de 1px)
            if 10 < count < 100:
                tous_les_ticks.append(int(current_y_sum / count))
            current_y_sum, count = 0, 0

    # --- FILTRAGE POUR 43 MARQUES ---
    # Structure : [2 Start] + [40 Lignes de questions] + [1 Final] = 43
    
    if len(tous_les_ticks) >= 43:
        # On renvoie une liste plate des 43 premières coordonnées Y
        return tous_les_ticks[:43]
    else:
        print(f"Attention : {len(tous_les_ticks)} marques trouvées au lieu de 43.")
        # On renvoie quand même ce qu'on a trouvé, ou une liste vide selon votre besoin
        return tous_les_ticks
    

def analyser_feuille(image_cv):
    """Analyse une image de grille QCM."""
    donnees_feuille = []
    
    # 1. Prétraitement OMR (pour les cases cochées)
    gris = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaire_omr = cv2.threshold(flou, 150, 255, cv2.THRESH_BINARY_INV)

    # 2. Extraction de l'en-tête (On envoie l'image COULEUR ici)
    num_etudiant, nom_prenom = extraire_texte(image_cv, *ZONE_IDENTITE)

    paires_y = detecter_lignes_y(binaire_omr)

    X_BLOCS = [115, 595, 1075, 1555, 2035] 
    DX_CASE = 52
    DX_REPENTIR = 65
    TAILLE_CASE = 28

    # 3. Extraction des QCM
    for bloc_idx, base_x in enumerate(X_BLOCS):
            for ligne_idx, (y_l1, y_l2) in enumerate(paires_y):
                q = (bloc_idx * 20) + (ligne_idx + 1)
                
                # --- LECTURE LIGNE 1 (Y_L1 détecté par la marque noire 1) ---
                res_l1 = [est_cochee(binaire_omr, base_x + (i*DX_CASE), y_l1, TAILLE_CASE, TAILLE_CASE) for i in range(5)]
                
                donnees_feuille.append({
                    'numéro étudiants': num_etudiant, 'nom': nom_prenom, 'question': q,
                    'ligne réponse': 1, 'repentance': False,
                    'Réponse A': res_l1[0], 'Réponse B': res_l1[1], 'Réponse C': res_l1[2], 
                    'Réponse D': res_l1[3], 'Réponse E': res_l1[4]
                })
                
                # --- LECTURE LIGNE 2 (Y_L2 détecté par la marque noire 2) ---
                # Lecture de la case de repentir (à gauche)
                repentir = est_cochee(binaire_omr, base_x - DX_REPENTIR, y_l2, TAILLE_CASE, TAILLE_CASE)
                
                # Lecture des cases A,B,C,D,E de la ligne 2
                res_l2 = [est_cochee(binaire_omr, base_x + (i*DX_CASE), y_l2, TAILLE_CASE, TAILLE_CASE) for i in range(5)]
                
                # On n'ajoute la ligne 2 au CSV que si elle a été utilisée
                if repentir or any(res_l2):
                    donnees_feuille.append({
                        'numéro étudiants': num_etudiant, 'nom': nom_prenom, 'question': q,
                        'ligne réponse': 2, 'repentance': repentir,
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