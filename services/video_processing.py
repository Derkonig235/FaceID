import cv2
from services.detection import detecter_visages, aligner_visage, preparer_image
from services.recognition import identifier_visage
from database.db import get_toutes_personnes

def traiter_video(chemin_video, chemin_sortie):
    """
    Traite une vidéo — détecte et reconnaît les visages sur chaque frame.
    
    Args:
        chemin_video: chemin vers la vidéo à analyser
        chemin_sortie: chemin où sauvegarder la vidéo annotée
    
    Returns:
        chemin_sortie: chemin de la vidéo résultat
    """
    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        return None

    # Récupérer les infos de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Préparer la sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(chemin_sortie, fourcc, fps, (width, height))

    # Charger la base de données une seule fois
    personnes = get_toutes_personnes()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Préparer l'image
        image_rgb = preparer_image(frame)

        # Détecter les visages
        faces = detecter_visages(image_rgb)

        for face in faces:
            crop, (x1, y1, x2, y2) = aligner_visage(image_rgb, face)
            if crop.size == 0:
                continue

            # Identifier
            nom, prenom, confiance = identifier_visage(crop, personnes)

            # Couleur selon reconnaissance
            couleur = (0, 255, 0) if nom != "Inconnu" else (0, 0, 255)

            # Dessiner
            cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
            label = f"{prenom} {nom} {confiance:.0f}%" if nom != "Inconnu" else "Inconnu"
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(label) * 12, y1), couleur, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    return chemin_sortie