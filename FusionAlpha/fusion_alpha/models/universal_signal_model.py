import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderSignalA(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        super(EncoderSignalA, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class EncoderSignalB(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        super(EncoderSignalB, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, proj_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class AdaptiveFusion(nn.Module):
    def __init__(self, latent_dim):
        super(AdaptiveFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2):
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)
        contradiction_score = 1.0 - cos_sim
        gate_input = contradiction_score.unsqueeze(1)
        gate_weight = self.gate(gate_input)
        fused = gate_weight * emb1 + (1 - gate_weight) * emb2
        return fused, contradiction_score, gate_weight

class DecisionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(DecisionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ContradictionLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(ContradictionLoss, self).__init__()
        self.weight = weight

    def forward(self, emb1, emb2, prediction):
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)
        contradiction_score = 1.0 - cos_sim
        confidence = torch.abs(prediction.view(-1))
        loss = self.weight * torch.mean(contradiction_score * (confidence ** 2))
        return loss

class UniversalSignalModel(nn.Module):
    """
    Universal signal model for contradiction detection across domains.
    
    Can be used for:
    - Healthcare: symptom vs. biomarker contradiction detection
    - Cybersecurity: behavior vs. baseline anomaly detection
    - Manufacturing: specification vs. measurement deviation detection
    - Media: claim vs. evidence contradiction detection
    - Finance: sentiment vs. price movement underhype detection
    """
    
    def __init__(self, signal_a_input_dim, signal_b_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim):
        super(UniversalSignalModel, self).__init__()
        self.encoder_signal_a = EncoderSignalA(signal_a_input_dim, encoder_hidden_dim, dropout_prob=0.1)
        self.encoder_signal_b = EncoderSignalB(signal_b_input_dim, encoder_hidden_dim, dropout_prob=0.1)
        self.projection = ProjectionHead(encoder_hidden_dim, proj_dim)
        self.adaptive_fusion = AdaptiveFusion(proj_dim)
        self.decision_head = DecisionHead(proj_dim, decision_hidden_dim, output_dim=1)

    def forward(self, signal_a_data, signal_b_data):
        signal_a_features = self.encoder_signal_a(signal_a_data)
        signal_b_features = self.encoder_signal_b(signal_b_data)
        proj_signal_a = self.projection(signal_a_features)
        proj_signal_b = self.projection(signal_b_features)
        fused, contradiction_score, gate_weight = self.adaptive_fusion(proj_signal_a, proj_signal_b)
        decision = self.decision_head(fused)
        return decision, contradiction_score, proj_signal_a, proj_signal_b, gate_weight

# Backward compatibility aliases
TradingModel = UniversalSignalModel
EncoderTechnical = EncoderSignalA
EncoderSentiment = EncoderSignalB