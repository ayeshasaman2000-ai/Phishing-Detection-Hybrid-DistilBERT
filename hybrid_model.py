import torch
import torch.nn as nn
from transformers import DistilBertModel

class HybridDistilBERT(nn.Module):
    def __init__(self, url_feature_dim=1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        input_dim = 768 + url_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, url_features):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        url_features = url_features.view(-1, 1)
        combined_features = torch.cat((cls_embeddings, url_features), dim=1)
        logits = self.classifier(self.dropout(combined_features))
        return logits.squeeze()
