import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')

def get_connection():
    """Retourne une connexion à la base de données."""
    return sqlite3.connect(DB_PATH)

def initialiser_db():
    """Crée la table personnes si elle n'existe pas encore."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS personnes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            prenom TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def ajouter_personne(nom, prenom, embedding):
    """Ajoute une personne avec son embedding dans la base."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO personnes (nom, prenom, embedding) VALUES (?, ?, ?)',
        (nom, prenom, embedding.tobytes())
    )
    conn.commit()
    conn.close()

def get_toutes_personnes():
    """Retourne toutes les personnes avec leurs embeddings."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, nom, prenom, embedding FROM personnes')
    rows = cursor.fetchall()
    conn.close()

    personnes = []
    for row in rows:
        id_, nom, prenom, embedding_bytes = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        personnes.append({
            'id': id_,
            'nom': nom,
            'prenom': prenom,
            'embedding': embedding
        })
    return personnes

def supprimer_personne(id_personne):
    """Supprime une personne par son id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM personnes WHERE id = ?', (id_personne,))
    conn.commit()
    conn.close()

def modifier_personne(id_personne, nom, prenom):
    """Modifie le nom et prénom d'une personne."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE personnes SET nom = ?, prenom = ? WHERE id = ?',
        (nom, prenom, id_personne)
    )
    conn.commit()
    conn.close()

# Initialiser la base au démarrage
initialiser_db()