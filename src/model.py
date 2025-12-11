import torch
from torch import nn
from transformers import DistilBertModel

class HateSpeechClassifier(nn.Module):
    def __init__(self, unfreeze_layers=6, dropout=0.4):
        super(HateSpeechClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for n, p in self.bert.named_parameters():
            p.requires_grad = False
        for param in self.bert.embeddings.parameters():
                param.requires_grad = True
        for layer in self.bert.transformer.layer[-unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.pooling_dropout = nn.Dropout(dropout * 0.3)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout*0.7),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout*0.4),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        last_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden = self.pooling_dropout(last_hidden)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        output = self.classifier(pooled)
        return output