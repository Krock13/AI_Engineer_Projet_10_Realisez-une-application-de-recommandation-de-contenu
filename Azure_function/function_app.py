import logging
import os
import json

# Azure Functions (Python v2) imports
import azure.functions as func
from azure.functions.decorators import FunctionApp, AuthLevel

import pickle
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# Instanciation de la FunctionApp (v2)
app = FunctionApp()

# Chargement des modèles / artefacts au démarrage
BASE_PATH = os.path.dirname(__file__)

with open(os.path.join(BASE_PATH, 'model_cf.pkl'), 'rb') as f:
    model_cf = pickle.load(f)

with open(os.path.join(BASE_PATH, 'model_cb.pkl'), 'rb') as f:
    model_cb = pickle.load(f)

with open(os.path.join(BASE_PATH, 'data_transform.pkl'), 'rb') as f:
    data_transform = pickle.load(f)

# Raccourcis CF
knn_model = model_cf['knn_model']               # NearestNeighbors
sparse_matrix = model_cf['sparse_matrix']       # shape = (nb_users, nb_articles_cf)
article_ids_cf = model_cf['article_ids']        # np.ndarray (taille nb_articles_cf)
user_id_to_index = model_cf['user_id_to_index'] # dict

# Raccourcis CB
user_profiles = model_cb['user_profiles']
articles_embeddings = model_cb['articles_embeddings'] # shape = (364047, emb_dim)

NUM_ARTICLES_GLOBAL = articles_embeddings.shape[0]  # 364047

# Fonctions utilitaires
def min_max_scale(scores: np.ndarray) -> np.ndarray:
    valid_mask = (scores > -9999)
    if not np.any(valid_mask):
        return scores
    valid_values = scores[valid_mask]
    mn = valid_values.min()
    mx = valid_values.max()
    if mn == mx:
        scores[valid_mask] = 1.0
        return scores
    rng = mx - mn
    scores[valid_mask] = (scores[valid_mask] - mn) / rng
    return scores

def predict_cf_knn_scores_subset(user_id: int, k_neighbors: int = 5) -> np.ndarray:
    """
    Score CF sur le 'subset' CF (nb_articles_cf).
    """
    nb_articles_cf = sparse_matrix.shape[1]
    scores_subset = np.zeros(nb_articles_cf, dtype=np.float32)
    if user_id not in user_id_to_index:
        return scores_subset

    user_idx = user_id_to_index[user_id]
    user_vector = sparse_matrix[user_idx]
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=k_neighbors+1)
    neighbor_indices = indices[0].tolist()
    if user_idx in neighbor_indices:
        neighbor_indices.remove(user_idx)
    neighbor_indices = neighbor_indices[:k_neighbors]

    neighbors_matrix = sparse_matrix[neighbor_indices]
    scores_subset = neighbors_matrix.sum(axis=0).A1

    # Exclure articles déjà vus
    already_viewed = user_vector.indices
    scores_subset[already_viewed] = -9999
    return scores_subset

def expand_cf_scores(scores_subset: np.ndarray) -> np.ndarray:
    """
    Etend le vecteur CF (taille nb_articles_cf) 
    vers un vecteur global (taille NUM_ARTICLES_GLOBAL).
    """
    scores_cf_global = np.full(NUM_ARTICLES_GLOBAL, -9999, dtype=np.float32)
    for col_idx, val in enumerate(scores_subset):
        art_global_id = article_ids_cf[col_idx]
        if 0 <= art_global_id < NUM_ARTICLES_GLOBAL:
            scores_cf_global[art_global_id] = val
    return scores_cf_global

def predict_cb_scores(user_id: int) -> np.ndarray:
    """
    Score CB (similarité) sur l'ensemble (364047 articles).
    """
    scores_cb = np.zeros(NUM_ARTICLES_GLOBAL, dtype=np.float32)
    if user_id not in user_profiles:
        return scores_cb
    user_emb = user_profiles[user_id].reshape(1, -1)
    sims = cosine_similarity(user_emb, articles_embeddings)[0]
    scores_cb = sims.astype(np.float32)
    return scores_cb

# Fonction de prédiction hybride
def predict_hybrid_recos(user_id: int, alpha: float = 0.5, top_n: int = 5, return_scores: bool = False):
    """
    Génère un ranking hybride pour TOUS les articles.
    
    - CF => vecteur de taille 364047 (après expand)
    - CB => vecteur de taille 364047
    - Exclusion des articles déjà vus en CB (si user_id connu en CF)
    - Normalisation min-max
    - scores_hybrid = alpha*CF + (1-alpha)*CB
    - tri décroissant => top_n

    Si return_scores=True, on retourne en plus la décomposition des scores 
    pour chaque article recommandé.
    """
    scores_cf_subset = predict_cf_knn_scores_subset(user_id, k_neighbors=5)
    scores_cf = expand_cf_scores(scores_cf_subset)  # shape(364047,)
    scores_cb = predict_cb_scores(user_id)          # shape(364047,)

    # Exclure articles déjà vus en CB
    if user_id in user_id_to_index:
        user_idx = user_id_to_index[user_id]
        user_vector = sparse_matrix[user_idx]
        already_viewed = user_vector.indices
        for col_idx in already_viewed:
            art_global_id = article_ids_cf[col_idx]
            if 0 <= art_global_id < NUM_ARTICLES_GLOBAL:
                scores_cb[art_global_id] = -9999

    # Normalisation
    scores_cf = min_max_scale(scores_cf)
    scores_cb = min_max_scale(scores_cb)

    # Combinaison
    scores_hybrid = alpha * scores_cf + (1 - alpha) * scores_cb

    # Tri
    sorted_indices = np.argsort(-scores_hybrid)
    top_indices = sorted_indices[:top_n]

    # Convertit en int
    final_recos = [int(i) for i in top_indices]

    if not return_scores:
        return final_recos

    recommendations_data = []
    for idx in top_indices:
        part_cf = alpha * scores_cf[idx]
        part_cb = (1 - alpha) * scores_cb[idx]
        total = scores_hybrid[idx]
        recommendations_data.append({
            "article_id": int(idx),
            "score_cf_part": float(part_cf),
            "score_cb_part": float(part_cb),
            "score_total": float(total)
        })
    return recommendations_data

# Function Azure - route="/recommend"
@app.route(route="recommend", auth_level=AuthLevel.FUNCTION)
def recommend_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Recommandation Hybride CF+CB : début")
    try:
        # Récup user_id
        user_id_str = req.params.get('user_id')
        if not user_id_str:
            try:
                body = req.get_json()
                user_id_str = body.get('user_id')
            except:
                pass
        if not user_id_str:
            return func.HttpResponse("Missing user_id", status_code=400)
        user_id = int(user_id_str)

        # Vérification si user_id existe
        if user_id not in user_id_to_index and user_id not in user_profiles:
            return func.HttpResponse(
                json.dumps({"error": f"L'utilisateur {user_id} est inconnu."}),
                status_code=404,
                mimetype="application/json"
            )

        # alpha
        alpha_str = req.params.get('alpha')
        alpha = float(alpha_str) if alpha_str else 0.5

        # Calcul
        final_recos = predict_hybrid_recos(user_id, alpha=alpha, top_n=5, return_scores=False)
        response_data = {
            "user_id": user_id,
            "recommendations": final_recos
            }

        return func.HttpResponse(
            status_code=200,
            body=json.dumps(response_data),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Erreur: {str(e)}", exc_info=True)
        return func.HttpResponse(
            status_code=500,
            body=json.dumps({"error": str(e)}),
            mimetype="application/json"
        )