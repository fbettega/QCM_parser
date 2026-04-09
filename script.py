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
ZONE_IDENTITE = (0.24, 0.05, 0.72, 0.12)

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

def extraire_texte(image_couleur,h, w, rel_x1, rel_y1, rel_x2, rel_y2):
    """Extrait le texte à partir de l'image BGR originale."""
    # Découpage du bloc identité
  

    x, x2 = int(rel_x1 * w), int(rel_x2 * w)
    y, y2 = int(rel_y1 * h), int(rel_y2 * h)

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

def detecter_lignes_y(binaire_omr, w):
    
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
        print(f"Attention : {len(tous_les_ticks)} marques trouvées au lieu de 43.")
        return tous_les_ticks
    
def detecter_colonnes_x(binaire_omr, w,paires_y):
    # 1. On définit la zone verticale globale des questions
    # On ignore les 2 premiers ticks (Start) et le dernier (Final) pour n'avoir que les questions
    y_start = paires_y[2][0] 
    y_end = paires_y[-1][0]
    
    zone_questions = binaire_omr[y_start:y_end, :int(w * 0.961) ]
    h_zone, w_zone = zone_questions.shape


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
                # Une colonne de questions sur ce type de document fait 
                # environ 15-18% de la largeur totale. On filtre les parasites < 100px.
                #if largeur > 100:
                paires_x.append((x_debut, x))
                en_colonne = False
                
        # Sécurité : on attend 5 colonnes
    if len(paires_x) != 5:
        print(f"Attention : {len(paires_x)} colonnes trouvées. Vérifiez le seuil.")
        
    return paires_x 

def analyser_feuille(image_cv):
    """Analyse une image de grille QCM."""
    donnees_feuille = []
    
    # 1. Prétraitement OMR (pour les cases cochées)
    gris = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaire_omr = cv2.threshold(flou, 150, 255, cv2.THRESH_BINARY_INV)
    h, w = image_cv.shape[:2]
    # 2. Extraction de l'en-tête (On envoie l'image COULEUR ici)
    num_etudiant, nom_prenom = extraire_texte(image_cv, h, w, *ZONE_IDENTITE)

    paires_y = detecter_lignes_y(binaire_omr, w)
    paires_x = detecter_colonnes_x(binaire_omr, w, paires_y)

    # début a 10 fin a 149
    # numéro question 13 a 39
    # bloc repantance 23 a 39
    # bloc de questions 44 a 60
    # espace entre les questions 61 a 65 


    # --- DEFINITION DES RATIOS (basés sur tes mesures) ---
    RATIO_REPENTIR = 0.151
    RATIO_A = 0.302
    RATIO_PAS = 0.151
    # Taille de la zone de lecture (environ 12% de la largeur du bloc)
    TAILLE_REL = 0.12

    # 3. Extraction des QCM
    # On parcourt les 5 blocs de colonnes
    for bloc_idx, (x_start, x_end) in enumerate(paires_x):
        # On parcourt les 20 questions du bloc
        # On commence à l'index 2 (saute les 2 start ticks) jusqu'à 41
        for q_in_col in range(20):
            q_numero = (bloc_idx * 20) + (q_in_col + 1)
            
            # Récupération des indices des deux lignes Y pour cette question
            idx_l1 = 2 + (q_in_col * 2)
            idx_l2 = idx_l1 + 1
            
            # Calcul du centre Y des deux lignes
            y_l1 = (paires_y[idx_l1][0] + paires_y[idx_l1][1]) // 2
            y_l2 = (paires_y[idx_l2][0] + paires_y[idx_l2][1]) // 2
            
            # Coordonnée X de base pour la case 'A'
            # On prend le début du bloc X détecté + un petit décalage pour être au centre du A
            base_x = x_start + OFFSET_A

            # --- LECTURE LIGNE 1 ---
            res_l1 = [
                est_cochee(binaire_omr, base_x + (i * DX_CASE), y_l1, TAILLE_CASE, TAILLE_CASE) 
                for i in range(5)
            ]
            
            donnees_feuille.append({
                'numéro étudiants': num_etudiant, 'nom': nom_prenom, 'question': q_numero,
                'ligne réponse': 1, 'repentance': False,
                'Réponse A': res_l1[0], 'Réponse B': res_l1[1], 'Réponse C': res_l1[2], 
                'Réponse D': res_l1[3], 'Réponse E': res_l1[4]
            })
            
            # --- LECTURE LIGNE 2 (Repentir) ---
            # Le repentir est à gauche de la case A
            x_rep = base_x - DX_REPENTIR
            repentir = est_cochee(binaire_omr, x_rep, y_l2, TAILLE_CASE, TAILLE_CASE)
            
            res_l2 = [
                est_cochee(binaire_omr, base_x + (i * DX_CASE), y_l2, TAILLE_CASE, TAILLE_CASE) 
                for i in range(5)
            ]
            
            # On n'ajoute la ligne 2 que si elle contient une marque
            if repentir or any(res_l2):
                donnees_feuille.append({
                    'numéro étudiants': num_etudiant, 'nom': nom_prenom, 'question': q_numero,
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