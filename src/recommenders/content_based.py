import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib


def load_news_text(news_csv_path, text_column='title'):
    news_df = pd.read_csv(news_csv_path)
    news_df.fillna("", inplace=True)
    news_ids = news_df['news_id'].tolist()
    texts = news_df[text_column].tolist()
    return news_ids, texts


def build_tfidf_embeddings(news_csv_path, output_vectorizer_path=None):
    news_ids, texts = load_news_text(news_csv_path)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    if output_vectorizer_path:
        os.makedirs(os.path.dirname(output_vectorizer_path), exist_ok=True)
        joblib.dump(vectorizer, output_vectorizer_path)

    return news_ids, tfidf_matrix, vectorizer


def recommend_top_n(user_history, tfidf_matrix, news_ids, top_n=5):
    if not user_history:
        return []

    news_id_to_index = {nid: idx for idx, nid in enumerate(news_ids)}
    valid_indices = [news_id_to_index[nid] for nid in user_history if nid in news_id_to_index]

    if not valid_indices:
        return []

    user_vector = tfidf_matrix[valid_indices].mean(axis=0)
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    recommended_indices = np.argsort(similarities)[::-1]

    recommended_ids = [news_ids[idx] for idx in recommended_indices if news_ids[idx] not in user_history]

    return recommended_ids[:top_n]


def save_tfidf_embeddings(tfidf_matrix, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(tfidf_matrix, path)


def load_tfidf_embeddings(path):
    return joblib.load(path)


def train_tfidf(news_df, text_column='title'):
    news_df.fillna("", inplace=True)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(news_df[text_column].tolist())
    return vectorizer, tfidf_matrix, news_df['news_id'].tolist()

def recommend_top_n(model, user_id, user_history, news_df, tfidf_matrix, vectorizer, top_n=5):
    news_ids = news_df['news_id'].tolist()
    news_id_to_index = {nid: idx for idx, nid in enumerate(news_ids)}
    history = user_history.get(user_id, [])

    valid_indices = [news_id_to_index[nid] for nid in history if nid in news_id_to_index]
    if not valid_indices:
        return []

    user_vector = tfidf_matrix[valid_indices].mean(axis=0)
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    recommended_indices = np.argsort(similarities)[::-1]
    recommended_ids = [news_ids[idx] for idx in recommended_indices if news_ids[idx] not in history]
    return recommended_ids[:top_n]