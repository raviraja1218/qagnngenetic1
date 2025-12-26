"""
QAGNN Phase 2: Neural network model for circuit prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CircuitPredictor(nn.Module):
    """
    Neural network to predict circuit accuracy from design parameters
    Architecture: 903 → 128 → 64 → 32 → 1
    """
    
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        # Input: 303 features (3 weights + 300 time points)
        self.fc1 = nn.Linear(303, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))  # Output between 0 and 1
        return x
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def test_model():
    """Test the model architecture"""
    model = CircuitPredictor()
    
    # Test forward pass
    batch_size = 32
    sample_input = torch.randn(batch_size, 303)
    output = model(sample_input)
    
    print(f'🎯 Model Architecture Test:')
    print(f'   Input shape: {sample_input.shape}')
    print(f'   Output shape: {output.shape}')
    print(f'   Output range: {output.min().item():.3f} - {output.max().item():.3f}')
    print(f'   Parameters: {model.get_num_parameters():,}')
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        sample_input = sample_input.cuda()
        output = model(sample_input)
        print(f'   GPU test passed: Output shape {output.shape}')
    
    return model

if __name__ == '__main__':
    test_model()
