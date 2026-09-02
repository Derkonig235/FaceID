import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from database.db import initialiser_db, ajouter_personne, get_toutes_personnes, supprimer_personne, modifier_personne
from services.detection import detecter_visages, aligner_visage, preparer_image
from services.recognition import extraire_embedding, identifier_visage
from services.video_processing import traiter_video

app = Flask(__name__)

# Configuration des dossiers
app.config['UPLOAD_IMAGES'] = 'uploads/images'
app.config['UPLOAD_VIDEOS'] = 'uploads/videos'
app.config['KNOWN_FACES'] = 'known_faces'
app.config['RESULTS'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max

# Initialiser la base de données au démarrage
initialiser_db()

# ─────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def page_image():
    return render_template('image.html')

@app.route('/video')
def page_video():
    return render_template('video.html')

@app.route('/live')
def page_live():
    return render_template('live.html')

@app.route('/persons')
def page_persons():
    personnes = get_toutes_personnes()
    return render_template('persons.html', personnes=personnes)

# ─────────────────────────────────────────
# API - GESTION DES PERSONNES
# ─────────────────────────────────────────

@app.route('/api/personnes', methods=['GET'])
def api_get_personnes():
    personnes = get_toutes_personnes()
    result = [{'id': p['id'], 'nom': p['nom'], 'prenom': p['prenom']} for p in personnes]
    return jsonify(result)

@app.route('/api/personnes/ajouter', methods=['POST'])
def api_ajouter_personne():
    nom = request.form.get('nom')
    prenom = request.form.get('prenom')
    photo = request.files.get('photo')

    if not nom or not prenom or not photo:
        return jsonify({'erreur': 'Nom, prénom et photo sont obligatoires'}), 400

    # Sauvegarder la photo
    chemin_photo = os.path.join(app.config['KNOWN_FACES'], photo.filename)
    photo.save(chemin_photo)

    # Lire et préparer l'image
    image_cv = cv2.imread(chemin_photo)
    image_rgb = preparer_image(image_cv)

    # Détecter le visage
    faces = detecter_visages(image_rgb)
    if not faces:
        return jsonify({'erreur': 'Aucun visage détecté sur la photo'}), 400

    # Prendre le visage avec la meilleure confiance
    face = max(faces, key=lambda f: f['confidence'])
    crop, _ = aligner_visage(image_rgb, face)

    # Extraire l'embedding
    embedding = extraire_embedding(crop)
    if embedding is None:
        return jsonify({'erreur': 'Impossible d\'extraire les caractéristiques du visage'}), 400

    # Sauvegarder dans la base
    ajouter_personne(nom, prenom, embedding)

    return jsonify({'message': f'{prenom} {nom} ajouté avec succès'})

@app.route('/api/personnes/modifier/<int:id_personne>', methods=['POST'])
def api_modifier_personne(id_personne):
    nom = request.form.get('nom')
    prenom = request.form.get('prenom')

    if not nom or not prenom:
        return jsonify({'erreur': 'Nom et prénom sont obligatoires'}), 400

    modifier_personne(id_personne, nom, prenom)
    return jsonify({'message': 'Personne modifiée avec succès'})

@app.route('/api/personnes/supprimer/<int:id_personne>', methods=['DELETE'])
def api_supprimer_personne(id_personne):
    supprimer_personne(id_personne)
    return jsonify({'message': 'Personne supprimée avec succès'})

# ─────────────────────────────────────────
# API - ANALYSE IMAGE
# ─────────────────────────────────────────

@app.route('/api/analyser/image', methods=['POST'])
def api_analyser_image():
    photo = request.files.get('photo')
    if not photo:
        return jsonify({'erreur': 'Aucune image reçue'}), 400

    # Sauvegarder l'image
    chemin = os.path.join(app.config['UPLOAD_IMAGES'], photo.filename)
    photo.save(chemin)

    # Lire et préparer
    image_cv = cv2.imread(chemin)
    image_rgb = preparer_image(image_cv)

    # Détecter les visages
    faces = detecter_visages(image_rgb)
    personnes = get_toutes_personnes()

    resultats = []
    for face in faces:
        crop, (x1, y1, x2, y2) = aligner_visage(image_rgb, face)
        if crop.size == 0:
            continue

        nom, prenom, confiance = identifier_visage(crop, personnes)
        couleur = (0, 255, 0) if nom != "Inconnu" else (0, 0, 255)

        # Dessiner sur l'image originale
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), couleur, 2)
        label = f"{prenom} {nom} {confiance:.0f}%" if nom != "Inconnu" else "Inconnu"
        cv2.rectangle(image_cv, (x1, y1 - 30), (x1 + len(label) * 12, y1), couleur, -1)
        cv2.putText(image_cv, label, (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        resultats.append({
            'nom': nom,
            'prenom': prenom,
            'confiance': round(confiance, 1)
        })

    # Sauvegarder le résultat
    nom_resultat = f"result_{photo.filename}"
    chemin_resultat = os.path.join(app.config['RESULTS'], nom_resultat)
    cv2.imwrite(chemin_resultat, image_cv)

    return jsonify({
        'nb_visages': len(resultats),
        'resultats': resultats,
        'image_resultat': f'/static/results/{nom_resultat}'
    })

# ─────────────────────────────────────────
# API - ANALYSE VIDÉO
# ─────────────────────────────────────────

@app.route('/api/analyser/video', methods=['POST'])
def api_analyser_video():
    video = request.files.get('video')
    if not video:
        return jsonify({'erreur': 'Aucune vidéo reçue'}), 400

    chemin_video = os.path.join(app.config['UPLOAD_VIDEOS'], video.filename)
    video.save(chemin_video)

    nom_resultat = f"result_{video.filename}"
    chemin_resultat = os.path.join(app.config['RESULTS'], nom_resultat)

    resultat = traiter_video(chemin_video, chemin_resultat)
    if resultat is None:
        return jsonify({'erreur': 'Impossible de traiter la vidéo'}), 400

    return jsonify({
        'message': 'Vidéo traitée avec succès',
        'video_resultat': f'/static/results/{nom_resultat}'
    })

if __name__ == '__main__':
    app.run(debug=True)