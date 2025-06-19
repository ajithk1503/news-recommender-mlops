import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score
import joblib
import os

def build_interaction_matrix(user_history_path: str, news_csv_path: str):
    with open(user_history_path, "r") as f:
        user_history = json.load(f)

    news_df = pd.read_csv(news_csv_path)
    news_ids = news_df['news_id'].unique().tolist()

    user_to_idx = {user: idx for idx, user in enumerate(user_history.keys())}
    news_to_idx = {news: idx for idx, news in enumerate(news_ids)}

    row_idx, col_idx, data = [], [], []
    for user, history in user_history.items():
        for news in history:
            if news in news_to_idx:
                row_idx.append(user_to_idx[user])
                col_idx.append(news_to_idx[news])
                data.append(1)

    num_users = len(user_to_idx)
    num_items = len(news_to_idx)

    matrix = csr_matrix((data, (row_idx, col_idx)), shape=(num_users, num_items))

    return matrix, user_to_idx, news_to_idx, news_ids


def train_nmf(click_matrix, n_components=50):
    nmf = NMF(n_components=n_components, init='nndsvd', random_state=42)
    user_factors = nmf.fit_transform(click_matrix)
    item_factors = nmf.components_
    return nmf, user_factors, item_factors


def recommend_top_n(user_index, user_factors, item_factors, news_ids, top_n=5):
    scores = np.dot(user_factors[user_index], item_factors)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [news_ids[i] for i in top_indices]


def evaluate_auc(click_matrix, predicted_scores, num_users_eval=100):
    scores = []
    for i in range(min(num_users_eval, click_matrix.shape[0])):
        true = click_matrix[i].toarray().flatten()
        pred = predicted_scores[i]
        if len(set(true)) > 1:
            try:
                auc = roc_auc_score(true, pred)
                scores.append(auc)
            except:
                continue
    return np.mean(scores) if scores else None


def save_model_and_embeddings(nmf_model, user_factors, item_factors,
                              model_path="models/latest_model.pkl",
                              user_emb_path="data/features/user_latent.npz",
                              item_emb_path="data/features/item_latent.npz",
                              registry_path="models/model_registry.json"):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(user_emb_path), exist_ok=True)

    joblib.dump(nmf_model, model_path)
    np.savez(user_emb_path, user_factors=user_factors)
    np.savez(item_emb_path, item_factors=item_factors)

    registry = {
        "model": "NMF_CollaborativeFiltering",
        "version": "v1",
        "n_components": user_factors.shape[1]
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)
