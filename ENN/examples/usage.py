"""
ENN Usage Examples - Domain-Agnostic Applications

Demonstrates ENN for diverse sequence modeling tasks:
- Audio processing (MFCC features, spectrograms)
- NLP (token sequences, embeddings)
- IoT sensors (time-series, anomaly detection)
- Scientific data (climate, biorhythms)
"""

import torch
import torch.nn as nn
from enn.config import Config
from enn.model import ENNModelWithSparsityControl as ENNModel
from enn.enhanced_model import create_attention_enn
from enn.bicep_adapter import create_bicep_enhanced_enn


def audio_processing_example():
    """ENN for audio feature processing (MFCC, spectrograms)."""
    config = Config()
    config.input_dim = 13  # MFCC features
    config.num_neurons = 12
    config.num_states = 6
    
    model = ENNModel(config)
    
    # Process audio features
    batch_size = 16
    time_frames = 100  # Audio time frames
    mfcc_features = torch.randn(batch_size, time_frames, config.input_dim)
    
    audio_output = model(mfcc_features)
    return audio_output[0]  # Return main output tensor


def nlp_processing_example():
    """ENN for natural language processing tasks."""
    config = Config()
    config.input_dim = 768  # BERT embedding dimension
    config.num_neurons = 16
    config.num_states = 8
    
    model = ENNModel(config)
    
    # Process token embeddings
    batch_size = 8
    sequence_length = 50  # Token sequence
    token_embeddings = torch.randn(batch_size, sequence_length, config.input_dim)
    
    text_output = model(token_embeddings)
    return text_output[0]  # Return main output tensor


def iot_sensor_example():
    """ENN for IoT sensor data and time-series analysis."""
    config = Config()
    config.input_dim = 6  # Multi-sensor readings (temp, humidity, pressure, etc.)
    config.num_neurons = 10
    config.num_states = 5
    
    model = ENNModel(config)
    
    # Process sensor time series
    batch_size = 32
    time_steps = 200  # Sensor readings over time
    sensor_data = torch.randn(batch_size, time_steps, config.input_dim)
    
    sensor_output = model(sensor_data)
    return sensor_output[0]  # Return main output tensor


def attention_variants_showcase():
    """Attention mechanisms for different application domains."""
    # Video analysis configuration
    video_config = Config()
    video_config.input_dim = 2048  # CNN feature maps
    
    # Climate data configuration  
    climate_config = Config()
    climate_config.input_dim = 15  # Weather variables
    
    # Bioinformatics configuration
    bio_config = Config()
    bio_config.input_dim = 4  # DNA/RNA nucleotides (A,T,G,C)
    
    # Minimal attention for real-time applications
    video_minimal = create_attention_enn(video_config, 'minimal')
    
    # Neuron-only for scientific data
    climate_neuron = create_attention_enn(climate_config, 'neuron_only')
    
    # Full attention for complex sequences
    bio_full = create_attention_enn(bio_config, 'full')
    
    # Test with domain-specific data shapes
    video_frames = torch.randn(4, 30, video_config.input_dim)  # 30 frames
    climate_series = torch.randn(16, 365, climate_config.input_dim)  # Daily data
    dna_sequence = torch.randn(8, 1000, bio_config.input_dim)  # Gene sequence
    
    video_output = video_minimal(video_frames)[0]  # Return main output
    climate_output = climate_neuron(climate_series)[0]  # Return main output
    bio_output = bio_full(dna_sequence)[0]  # Return main output
    
    return video_output, climate_output, bio_output


def bicep_integration():
    """Integration with BICEP stochastic dynamics."""
    config = Config()
    config.input_dim = 5
    config.num_neurons = 10
    config.num_states = 5
    
    try:
        # Create BICEP-enhanced ENN
        model = create_bicep_enhanced_enn(config, integration_mode='adapter')
        
        # Process with stochastic dynamics
        input_data = torch.randn(16, 20, config.input_dim)
        output = model(input_data)
        
        return output[0], {"bicep_enhanced": True}
    except Exception as e:
        print(f"BICEP integration using fallback due to: {e}")
        # Fallback to regular ENN
        model = ENNModel(config)
        input_data = torch.randn(16, 20, config.input_dim)
        output = model(input_data)
        return output[0], {"bicep_enhanced": False, "fallback": True}


def scientific_applications_example():
    """Training ENN on scientific and research data."""
    # Seismic analysis configuration
    seismic_config = Config()
    seismic_config.input_dim = 8  # Seismometer channels
    seismic_config.output_dim = 3  # Earthquake magnitude, depth, location
    seismic_config.learning_rate = 0.0005
    seismic_config.batch_size = 16
    seismic_config.epochs = 50
    
    model = create_attention_enn(seismic_config, 'full')
    
    # Setup training for seismic data
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=seismic_config.learning_rate,
        weight_decay=1e-4
    )
    
    print("Training ENN for seismic analysis...")
    for epoch in range(seismic_config.epochs):
        # Simulate seismic waveform data
        waveforms = torch.randn(seismic_config.batch_size, 300, seismic_config.input_dim)
        earthquake_params = torch.randn(seismic_config.batch_size, seismic_config.output_dim)
        
        optimizer.zero_grad()
        predictions = model(waveforms)[0]  # Get main output
        # Reshape predictions to match target shape
        if predictions.dim() > 2:
            predictions = predictions.mean(dim=1)  # Average over sequence dimension
        # Handle batch size mismatch
        target_batch_size = earthquake_params.size(0)
        if predictions.size(0) != target_batch_size:
            predictions = predictions[:target_batch_size]  # Trim batch dimension
        if predictions.size(-1) != earthquake_params.size(-1):
            predictions = predictions[:, :earthquake_params.size(-1)]  # Trim to target size
        loss = criterion(predictions, earthquake_params)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Seismic Analysis Loss: {loss.item():.6f}")
    
    return model


def anomaly_detection_example():
    """ENN for real-time anomaly detection in sensor networks."""
    config = Config()
    config.input_dim = 12  # Multiple sensor types
    config.output_dim = 1   # Anomaly score
    config.learning_rate = 0.001
    config.batch_size = 64
    config.epochs = 30
    
    model = create_attention_enn(config, 'minimal')  # Lightweight for real-time
    
    criterion = nn.BCEWithLogitsLoss()  # Binary anomaly classification
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print("Training ENN for anomaly detection...")
    for epoch in range(config.epochs):
        # Simulate sensor network data
        sensor_readings = torch.randn(config.batch_size, 100, config.input_dim)
        anomaly_labels = torch.randint(0, 2, (config.batch_size, config.output_dim)).float()
        
        optimizer.zero_grad()
        anomaly_scores = model(sensor_readings)[0]  # Get main output
        # Reshape to match target shape
        if anomaly_scores.dim() > 2:
            anomaly_scores = anomaly_scores.mean(dim=1)  # Average over sequence dimension
        # Handle batch size mismatch
        target_batch_size = anomaly_labels.size(0)
        if anomaly_scores.size(0) != target_batch_size:
            anomaly_scores = anomaly_scores[:target_batch_size]  # Trim batch dimension
        if anomaly_scores.size(-1) != anomaly_labels.size(-1):
            anomaly_scores = anomaly_scores[:, :anomaly_labels.size(-1)]  # Trim to target size
        loss = criterion(anomaly_scores, anomaly_labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Anomaly Detection Loss: {loss.item():.6f}")
    
    return model


if __name__ == "__main__":
    print("=== ENN Domain-Agnostic Applications ===")
    
    print("\n1. Audio Processing Example...")
    audio_output = audio_processing_example()
    print(f"Audio analysis output shape: {audio_output.shape}")
    
    print("\n2. NLP Processing Example...")
    text_output = nlp_processing_example()
    print(f"Text analysis output shape: {text_output.shape}")
    
    print("\n3. IoT Sensor Example...")
    sensor_output = iot_sensor_example()
    print(f"Sensor analysis output shape: {sensor_output.shape}")
    
    print("\n4. Attention Variants for Different Domains...")
    video_out, climate_out, bio_out = attention_variants_showcase()
    print(f"Video analysis output: {video_out.shape}")
    print(f"Climate analysis output: {climate_out.shape}")
    print(f"Bioinformatics output: {bio_out.shape}")
    
    print("\n5. BICEP Stochastic Integration...")
    bicep_output, stochastic_info = bicep_integration()
    print(f"BICEP enhanced output: {bicep_output.shape}")
    
    print("\n6. Scientific Applications...")
    seismic_model = scientific_applications_example()
    
    print("\n7. Real-time Anomaly Detection...")
    anomaly_model = anomaly_detection_example()
    
    print("\n=== ENN successfully demonstrated across multiple domains ===")