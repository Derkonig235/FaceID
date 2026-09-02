import cv2
import numpy as np
from mtcnn import MTCNN

# Initialiser le détecteur une seule fois
detector = MTCNN()

def detecter_visages(image_rgb):
    """
    Détecte les visages dans une image RGB.
    
    Args:
        image_rgb: image en format RGB (numpy array)
    
    Returns:
        Liste des visages détectés avec leurs coordonnées et points clés
    """
    faces = detector.detect_faces(image_rgb)
    return faces

def aligner_visage(image_rgb, face):
    """
    Aligne un visage détecté pour améliorer la reconnaissance.
    
    Args:
        image_rgb: image en format RGB
        face: visage détecté par MTCNN
    
    Returns:
        crop: image du visage aligné
        coordonnées: (x1, y1, x2, y2)
    """
    keypoints = face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Calculer l'angle de rotation
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    centre = (
        float((left_eye[0] + right_eye[0]) / 2),
        float((left_eye[1] + right_eye[1]) / 2)
    )

    # Rotation
    h_img, w_img = image_rgb.shape[:2]
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    image_alignee = cv2.warpAffine(image_rgb, M, (w_img, h_img))

    # Extraire le crop avec marge
    x, y, w, h = face['box']
    x = max(0, x)
    y = max(0, y)
    marge = int(max(w, h) * 0.2)
    x1 = max(0, x - marge)
    y1 = max(0, y - marge)
    x2 = min(w_img, x + w + marge)
    y2 = min(h_img, y + h + marge)

    return image_alignee[y1:y2, x1:x2], (x1, y1, x2, y2)

def preparer_image(image_cv):
    """
    Prépare une image OpenCV pour la détection.
    Redimensionne si trop grande et convertit en RGB.
    
    Args:
        image_cv: image en format BGR (OpenCV)
    
    Returns:
        image_rgb: image convertie en RGB
    """
    h_img, w_img = image_cv.shape[:2]
    max_dim = 1280
    if max(w_img, h_img) > max_dim:
        scale = max_dim / max(w_img, h_img)
        image_cv = cv2.resize(image_cv, (int(w_img * scale), int(h_img * scale)))

    return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)