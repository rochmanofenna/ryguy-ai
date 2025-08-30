import pytest
import torch
import torch.nn as nn
from typing import Dict

from enn.pushout import ContextCollapseHead, DeterministicContextHead


class TestConfig:
    """Test configuration object."""
    def __init__(self):
        self.num_layers = 2
        self.num_neurons = 64
        self.num_states = 8
        self.decay_rate = 0.01
        self.recency_factor = 0.9
        self.buffer_size = 10
        self.low_power_k = 4
        self.sparsity_threshold = 0.1
        self.compressed_dim = 32
        self.input_dim = 128
        self.context_dim = 64
        self.dropout = 0.1


class TestContextCollapseHead:
    """Test suite for the context collapse head."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 64
        self.num_states = 8
        self.context_dim = 32
        self.batch_size = 4
        
        self.head = ContextCollapseHead(
            input_dim=self.input_dim,
            num_states=self.num_states,
            context_dim=self.context_dim,
            dropout=0.0  # Disable dropout for deterministic tests
        )
        
        # Create test data with known contradiction patterns
        self.test_states = torch.randn(self.batch_size, self.input_dim, self.num_states)
        
        # Create contradictory states (alpha/not-alpha pattern)
        self.contradictory_states = torch.zeros(self.batch_size, self.input_dim, self.num_states)
        self.contradictory_states[:, :, 0] = 1.0   # alpha state
        self.contradictory_states[:, :, 1] = -1.0  # not-alpha state
        
        # Create harmonious states (all similar)
        self.harmonious_states = torch.ones(self.batch_size, self.input_dim, self.num_states) * 0.5
        
    def test_output_shapes(self):
        """Test that output tensors have correct shapes."""
        p_t, contradiction_score, diagnostics = self.head(
            self.test_states, 
            return_diagnostics=True
        )
        
        # Check shapes
        assert p_t.shape == (self.batch_size, self.input_dim, self.context_dim)
        assert contradiction_score.shape == (self.batch_size, self.input_dim)
        assert isinstance(diagnostics, dict)
        
        # Check diagnostic keys
        expected_keys = {'entropy', 'interference', 'attention_weights', 'states_norm'}
        assert set(diagnostics.keys()) == expected_keys
        
    def test_contradiction_detection(self):
        """Test that contradictory states produce higher contradiction scores."""
        # Test contradictory states
        _, contra_score, _ = self.head(self.contradictory_states)
        
        # Test harmonious states  
        _, harmony_score, _ = self.head(self.harmonious_states)
        
        # Contradictory states should have higher scores
        assert contra_score.mean() > harmony_score.mean()
        
    def test_deterministic_behavior(self):
        """Test that the same input produces the same output."""
        # Set model to eval mode for deterministic behavior
        self.head.eval()
        
        with torch.no_grad():
            p_t1, score1, _ = self.head(self.test_states)
            p_t2, score2, _ = self.head(self.test_states)
            
            # Results should be identical
            torch.testing.assert_close(p_t1, p_t2, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(score1, score2, rtol=1e-5, atol=1e-6)
            
    def test_gradient_flow(self):
        """Test that gradients flow properly through the head."""
        p_t, contradiction_score, _ = self.head(self.test_states)
        
        # Create a dummy loss
        loss = p_t.mean() + contradiction_score.mean()
        loss.backward()
        
        # Check that parameters have gradients
        for param in self.head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            
    def test_entropy_computation(self):
        """Test entropy computation for known patterns."""
        # All zeros should have low entropy
        zero_states = torch.zeros(1, 1, self.num_states)
        entropy_zero = self.head.compute_state_entropy(zero_states)
        
        # Uniform distribution should have high entropy
        uniform_states = torch.ones(1, 1, self.num_states) / self.num_states
        entropy_uniform = self.head.compute_state_entropy(uniform_states)
        
        # Uniform should have higher entropy than zeros
        assert entropy_uniform > entropy_zero
        
    def test_batch_consistency(self):
        """Test that processing batches vs individual samples is consistent."""
        # Process full batch
        p_t_batch, score_batch, _ = self.head(self.test_states)
        
        # Process individual samples
        p_t_individual = []
        score_individual = []
        
        for i in range(self.batch_size):
            p_t_i, score_i, _ = self.head(self.test_states[i:i+1])
            p_t_individual.append(p_t_i)
            score_individual.append(score_i)
            
        p_t_individual = torch.cat(p_t_individual, dim=0)
        score_individual = torch.cat(score_individual, dim=0)
        
        # Results should be close (allowing for small numerical differences)
        torch.testing.assert_close(p_t_batch, p_t_individual, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(score_batch, score_individual, rtol=1e-4, atol=1e-5)


class TestDeterministicContextHead:
    """Test suite for the deterministic context head."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 32
        self.num_states = 4
        self.context_dim = 16
        self.batch_size = 2
        
        self.det_head = DeterministicContextHead(
            input_dim=self.input_dim,
            num_states=self.num_states,
            context_dim=self.context_dim
        )
        
        # Create known test patterns
        self.test_states = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0]] * self.input_dim,  # Low variance
            [[1.0, 10.0, 1.0, 10.0]] * self.input_dim  # High variance
        ], dtype=torch.float32)
        
    def test_deterministic_output(self):
        """Test that deterministic head produces expected outputs."""
        p_t, contradiction_score, diagnostics = self.det_head(
            self.test_states, 
            return_diagnostics=True
        )
        
        # Check shapes
        assert p_t.shape == (self.batch_size, self.input_dim, self.context_dim)
        assert contradiction_score.shape == (self.batch_size, self.input_dim)
        
        # High variance sample should have higher contradiction score
        assert contradiction_score[1, 0] > contradiction_score[0, 0]
        
        # p_t should be the mean repeated across context dimension
        expected_mean_0 = self.test_states[0, 0].mean()
        expected_mean_1 = self.test_states[1, 0].mean()
        
        assert torch.allclose(p_t[0, 0], torch.full((self.context_dim,), expected_mean_0))
        assert torch.allclose(p_t[1, 0], torch.full((self.context_dim,), expected_mean_1))
        
    def test_perfect_determinism(self):
        """Test that deterministic head gives exactly the same output."""
        p_t1, score1, _ = self.det_head(self.test_states)
        p_t2, score2, _ = self.det_head(self.test_states)
        
        # Should be exactly equal
        assert torch.equal(p_t1, p_t2)
        assert torch.equal(score1, score2)


class TestENNModelIntegration:
    """Test the integration with ENN model."""
    
    def setup_method(self):
        """Setup ENN model for testing."""
        from enn.model import ENNModelWithSparsityControl
        
        self.cfg = TestConfig()
        self.model = ENNModelWithSparsityControl(self.cfg)
        
        # Test input data
        self.batch_size = 2
        self.time_steps = 10
        self.features = self.cfg.input_dim
        self.test_input = torch.randn(self.batch_size, self.time_steps, self.features)
        
    def test_model_with_p_t(self):
        """Test that model returns p_t correctly."""
        logits, p_t, contradiction_score, diagnostics = self.model(
            self.test_input,
            return_p_t=True,
            return_diagnostics=True
        )
        
        # Check that p_t is returned
        assert p_t is not None
        assert contradiction_score is not None
        assert diagnostics is not None
        
        # Check shapes
        assert p_t.shape == (self.batch_size, self.cfg.num_neurons, self.cfg.context_dim)
        assert contradiction_score.shape == (self.batch_size, self.cfg.num_neurons)
        
    def test_model_without_p_t(self):
        """Test that model can work without p_t for backward compatibility."""
        logits, p_t, contradiction_score, diagnostics = self.model(
            self.test_input,
            return_p_t=False
        )
        
        # p_t should be None when not requested
        assert p_t is None
        assert contradiction_score is None
        assert diagnostics is None
        
        # But logits should still be returned
        assert logits is not None
        assert logits.shape[0] == self.batch_size
        
    def test_p_t_gradient_flow(self):
        """Test that p_t gradients flow through the entire model."""
        logits, p_t, contradiction_score, _ = self.model(
            self.test_input,
            return_p_t=True
        )
        
        # Create loss from p_t
        p_t_loss = p_t.mean()
        p_t_loss.backward(retain_graph=True)
        
        # Check that pushout head parameters have gradients
        for param in self.model.pushout_head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


if __name__ == "__main__":
    # Run a quick smoke test
    pytest.main([__file__, "-v"])