from river import compose
from river import feature_extraction
from river import linear_model
from river import metrics
from river import preprocessing
import json
import pandas as pd


class OnlineNewsRecommender:
    def __init__(self):
        self.model = compose.Pipeline(
            ('vectorizer', feature_extraction.BagOfWords(on='text')),
            ('tfidf', feature_extraction.TFIDF()),
            ('model', linear_model.LogisticRegression())
        )
        self.metric = metrics.Accuracy()

    def learn_one(self, text, label):
        self.model.learn_one({'text': text}, label)

    def predict_proba_one(self, text):
        return self.model.predict_proba_one({'text': text})

    def predict_one(self, text):
        return self.model.predict_one({'text': text})

    def update_metric(self, text, label):
        y_pred = self.predict_one(text)
        self.metric.update(label, y_pred)

    def get_metric(self):
        return self.metric.get()

    def recommend_top_k(self, user_history_texts, candidate_news_df, k=5):
        if not user_history_texts:
            return candidate_news_df['news_id'].head(k).tolist()

        recommendations = []
        for _, row in candidate_news_df.iterrows():
            text = row['title'] + " " + row.get('abstract', '')
            score = self.predict_proba_one(text).get(True, 0.0)
            recommendations.append((row['news_id'], score))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in recommendations[:k]]


def load_user_history_json(path):
    with open(path) as f:
        return json.load(f)


def load_news_df(path):
    return pd.read_csv(path)
