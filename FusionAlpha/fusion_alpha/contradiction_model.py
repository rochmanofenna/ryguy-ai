import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Component 1: EncoderSignalA (Universal Primary Signal Encoder)
# ---------------------------
class EncoderSignalA(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        """
        Universal encoder for primary signals across domains:
        - Finance: OHLCV + technical indicators (SMA, EMA, RSI, MACD, etc.)
        - Healthcare: patient-reported symptom scores  
        - Cybersecurity: stated user behavior patterns
        - Manufacturing: design specifications and target values
        - Media: claim strength and confidence scores
        """
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

# ---------------------------
# Component 2: EncoderSignalB (Universal Reference Signal Encoder)
# ---------------------------
class EncoderSignalB(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        """
        Universal encoder for reference/baseline signals across domains:
        - Finance: sentiment embeddings (FinBERT or similar)
        - Healthcare: objective biomarker levels and test results
        - Cybersecurity: actual network behavior and system logs
        - Manufacturing: actual measurements and quality control data
        - Media: evidence support scores and fact-check results
        """
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

# Backward compatibility aliases
EncoderTechnical = EncoderSignalA
EncoderSentiment = EncoderSignalB

# ---------------------------
# Component 3: ProjectionHead
# ---------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        """
        Projects features from an encoder into a shared latent space.
        """
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, proj_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

# ---------------------------
# Component 4: AdaptiveFusion
# ---------------------------
class AdaptiveFusion(nn.Module):
    def __init__(self, latent_dim):
        """
        Fuses the projected embeddings using a gating mechanism.
        The gate network uses the cosine similarity-based contradiction score.
        """
        super(AdaptiveFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Produces a weight between 0 and 1.
        )
        
    def forward(self, emb1, emb2):
        # Compute cosine similarity between the two embeddings.
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)  # Shape: (batch,)
        contradiction_score = 1.0 - cos_sim  # Higher value indicates more divergence.
        
        # Process contradiction score through the gating network.
        gate_input = contradiction_score.unsqueeze(1)  # Shape: (batch, 1)
        gate_weight = self.gate(gate_input)  # Weight for emb1; shape: (batch, 1)
        
        # Fuse the embeddings using the gate weight:
        # fused = weight * emb1 + (1 - weight) * emb2.
        fused = gate_weight * emb1 + (1 - gate_weight) * emb2
        return fused, contradiction_score, gate_weight

# ---------------------------
# Component 5: DecisionHead (FusionNet)
# ---------------------------
class DecisionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        """
        Takes the fused latent representation and outputs a scalar prediction.
        """
        super(DecisionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ---------------------------
# Component 6: ContradictionLoss Module
# ---------------------------
class ContradictionLoss(nn.Module):
    def __init__(self, weight=1.0):
        """
        Penalizes high-confidence predictions when the projected embeddings diverge.
        """
        super(ContradictionLoss, self).__init__()
        self.weight = weight
        
    def forward(self, emb1, emb2, prediction):
        # Compute cosine similarity and derive contradiction score.
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)
        contradiction_score = 1.0 - cos_sim  # Shape: (batch,)
        # Use the absolute value of the prediction as a proxy for confidence.
        confidence = torch.abs(prediction.view(-1))
        loss = self.weight * torch.mean(contradiction_score * (confidence ** 2))
        return loss

# ---------------------------
# Component 7: UniversalSignalModel (Universal Contradiction Detection Model)
# ---------------------------
class UniversalSignalModel(nn.Module):
    def __init__(self, signal_a_input_dim, signal_b_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim):
        """
        Universal end-to-end model for contradiction detection across domains:
          1. Encode primary and reference signals from any domain
          2. Project into a shared latent space
          3. Fuse adaptively using a contradiction-aware gate
          4. Produce a final decision output
          
        Domain applications:
        - Finance: technical vs. sentiment contradiction (underhype detection)
        - Healthcare: symptom vs. biomarker contradiction detection
        - Cybersecurity: stated vs. actual behavior anomaly detection
        - Manufacturing: specification vs. measurement deviation detection
        - Media: claim vs. evidence contradiction detection
        """
        super(UniversalSignalModel, self).__init__()
        self.encoder_signal_a = EncoderSignalA(signal_a_input_dim, encoder_hidden_dim, dropout_prob=0.1)
        self.encoder_signal_b = EncoderSignalB(signal_b_input_dim, encoder_hidden_dim, dropout_prob=0.1)
        self.projection = ProjectionHead(encoder_hidden_dim, proj_dim)
        self.adaptive_fusion = AdaptiveFusion(proj_dim)
        self.decision_head = DecisionHead(proj_dim, decision_hidden_dim, output_dim=1)
        
    def forward(self, signal_a_data, signal_b_data):
        # Step 1: Encode each signal modality
        signal_a_features = self.encoder_signal_a(signal_a_data)
        signal_b_features = self.encoder_signal_b(signal_b_data)
        
        # Step 2: Project to shared latent space
        proj_signal_a = self.projection(signal_a_features)
        proj_signal_b = self.projection(signal_b_features)
        
        # Step 3: Fuse using adaptive fusion with contradiction awareness
        fused, contradiction_score, gate_weight = self.adaptive_fusion(proj_signal_a, proj_signal_b)
        
        # Step 4: Get final contradiction decision
        decision = self.decision_head(fused)
        return decision, contradiction_score, proj_signal_a, proj_signal_b, gate_weight

# Backward compatibility wrapper for financial trading applications
class TradingModel(UniversalSignalModel):
    def __init__(self, tech_input_dim, sentiment_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim):
        """Backward compatibility wrapper for financial trading use case"""
        super().__init__(tech_input_dim, sentiment_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim)
        # Aliases for backward compatibility
        self.encoder_tech = self.encoder_signal_a
        self.encoder_sent = self.encoder_signal_b
        
    def forward(self, tech_data, sentiment_data):
        """Backward compatibility forward method"""
        return super().forward(tech_data, sentiment_data)

# ---------------------------
# Testing and Training Loop Scaffold
# ---------------------------
if __name__ == '__main__':
    # Define dimensions for dummy data across different domains
    batch_size = 8
    signal_a_input_dim = 32     # Universal primary signal dimension
    signal_b_input_dim = 768    # Universal reference signal dimension
    encoder_hidden_dim = 64
    proj_dim = 32
    decision_hidden_dim = 64
    
    # Create dummy inputs
    signal_a_data = torch.randn(batch_size, signal_a_input_dim)
    signal_b_data = torch.randn(batch_size, signal_b_input_dim)
    # Dummy target: scalar contradiction score for each sample
    target = torch.randn(batch_size, 1)
    
    # Instantiate the universal model and loss modules
    model = UniversalSignalModel(signal_a_input_dim, signal_b_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim)
    prediction_loss_fn = nn.MSELoss()
    contradiction_loss_fn = ContradictionLoss(weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Universal Contradiction Detection Model Test")
    print("=" * 50)
    
    # Training loop (using dummy data)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        decision, contradiction_score, proj_signal_a, proj_signal_b, gate_weight = model(signal_a_data, signal_b_data)
        
        # Compute primary prediction loss
        primary_loss = prediction_loss_fn(decision, target)
        # Compute contradiction regularization loss
        contr_loss = contradiction_loss_fn(proj_signal_a, proj_signal_b, decision)
        
        total_loss = primary_loss + contr_loss
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Total Loss = {total_loss.item():.4f}, Primary Loss = {primary_loss.item():.4f}, Contradiction Loss = {contr_loss.item():.4f}")
    
    # Testing a forward pass
    model.eval()
    with torch.no_grad():
        decision, contradiction_score, proj_signal_a, proj_signal_b, gate_weight = model(signal_a_data, signal_b_data)
        print("\nSample Results:")
        print(f"Decision Output: {decision[:3].squeeze()}")
        print(f"Contradiction Score: {contradiction_score[:3]}")
        print(f"Gate Weights: {gate_weight[:3].squeeze()}")
        
    # Test backward compatibility with financial model
    print("\nBackward Compatibility Test (Financial Model):")
    financial_model = TradingModel(signal_a_input_dim, signal_b_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim)
    with torch.no_grad():
        decision, contradiction_score, proj_tech, proj_sent, gate_weight = financial_model(signal_a_data, signal_b_data)
        print(f"Financial Model Decision: {decision[:3].squeeze()}")
        print(f"Financial Model Contradiction: {contradiction_score[:3]}")
        
    print("\nModel successfully supports universal contradiction detection across domains!")