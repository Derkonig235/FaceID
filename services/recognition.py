import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image as PILImage

# Initialiser le modèle une seule fois
model = InceptionResnetV1(pretrained='vggface2').eval()

def extraire_embedding(crop_rgb):
    """
    Extrait l'embedding d'un visage aligné.
    
    Args:
        crop_rgb: image du visage en RGB (numpy array)
    
    Returns:
        embedding: vecteur de caractéristiques du visage (numpy array)
        ou None si erreur
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return None

    img = PILImage.fromarray(crop_rgb).resize((160, 160))
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding[0].numpy()

def distance_cosinus(vec1, vec2):
    """
    Calcule la distance cosinus entre deux embeddings.
    Plus la distance est faible, plus les visages sont similaires.
    """
    dot = np.dot(vec1, vec2)
    norme = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return 1 - (dot / norme) if norme != 0 else 1.0

def identifier_visage(crop_rgb, personnes, seuil=0.5):
    """
    Identifie un visage en le comparant à la base de données.
    
    Args:
        crop_rgb: image du visage en RGB
        personnes: liste des personnes depuis la base de données
        seuil: distance maximale pour considérer une correspondance
    
    Returns:
        nom: nom de la personne reconnue ou "Inconnu"
        prenom: prénom de la personne reconnue ou ""
        confiance: pourcentage de confiance
    """
    embedding_inconnu = extraire_embedding(crop_rgb)
    if embedding_inconnu is None:
        return "Inconnu", "", 0.0

    meilleur_nom = "Inconnu"
    meilleur_prenom = ""
    meilleure_distance = seuil

    for personne in personnes:
        distance = distance_cosinus(embedding_inconnu, personne['embedding'])
        if distance < meilleure_distance:
            meilleure_distance = distance
            meilleur_nom = personne['nom']
            meilleur_prenom = personne['prenom']

    confiance = max(0, (1 - meilleure_distance) * 100)
    return meilleur_nom, meilleur_prenom, confiance