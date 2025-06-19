import torch
import torch.nn as nn
import torch.nn.functional as F

class NewsClickPredictor(nn.Module):
    def __init__(self, user_dim, news_dim, hidden_dim=64):
        super(NewsClickPredictor, self).__init__()
        self.user_embedding = nn.Linear(user_dim, hidden_dim)
        self.news_embedding = nn.Linear(news_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim * 2, 1)

    def forward(self, user_vec, news_vec):
        user_repr = F.relu(self.user_embedding(user_vec))
        news_repr = F.relu(self.news_embedding(news_vec))
        concat = torch.cat([user_repr, news_repr], dim=1)
        out = torch.sigmoid(self.output(concat))
        return out

def get_model(user_dim, news_dim):
    return NewsClickPredictor(user_dim, news_dim)
