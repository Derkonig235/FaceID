import cv2
from services.detection import detecter_visages, aligner_visage, preparer_image
from services.recognition import identifier_visage
from database.db import get_toutes_personnes

def analyser_video_stream(chemin_video, chemin_sortie):
    """
    Analyse une vidéo frame par frame et yield les résultats au fur et à mesure.
    """
    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        yield {'erreur': 'Impossible d\'ouvrir la vidéo'}
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(chemin_sortie, fourcc, fps, (width, height))

    personnes = get_toutes_personnes()
    frame_index = 0
    stats = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_index / fps, 2)
        image_rgb = preparer_image(frame)
        
        h_orig, w_orig = frame.shape[:2]
        h_prep, w_prep = image_rgb.shape[:2]
        scale_x = w_orig / w_prep
        scale_y = h_orig / h_prep

        faces = detecter_visages(image_rgb)
        detections = []

        for face in faces:
            crop, (x1, y1, x2, y2) = aligner_visage(image_rgb, face)
            if crop.size == 0:
                continue

            nom, prenom, confiance = identifier_visage(crop, personnes)
            couleur = (0, 255, 0) if nom != "Inconnu" else (0, 0, 255)

            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)

            cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), couleur, 2)
            label = f"{prenom} {nom} {confiance:.0f}%" if nom != "Inconnu" else "Inconnu"
            cv2.rectangle(frame, (x1_orig, y1_orig - 30), (x1_orig + len(label) * 12, y1_orig), couleur, -1)
            cv2.putText(frame, label, (x1_orig + 4, y1_orig - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            detections.append({
                'nom': nom,
                'prenom': prenom,
                'confiance': float(round(confiance, 1)),
                'timestamp': timestamp
            })

            # Statistiques
            cle = f"{prenom} {nom}" if nom != "Inconnu" else "Inconnu"
            if cle not in stats:
                stats[cle] = {'count': 0, 'confiance_total': 0}
            stats[cle]['count'] += 1
            stats[cle]['confiance_total'] += confiance

        out.write(frame)

        progression = round((frame_index / total_frames) * 100, 1)
        yield {
            'type': 'frame',
            'frame': frame_index,
            'total': total_frames,
            'progression': progression,
            'timestamp': timestamp,
            'detections': detections
        }

        frame_index += 1

    cap.release()
    out.release()

    # Résumé final
    resume = []
    for cle, data in stats.items():
        resume.append({
            'label': cle,
            'apparitions': data['count'],
            'confiance_moyenne': float(round(data['confiance_total'] / data['count'], 1))
        })

    yield {
        'type': 'termine',
        'resume': resume,
        'video_resultat': f'/static/results/{chemin_sortie.split("/")[-1]}'
    }