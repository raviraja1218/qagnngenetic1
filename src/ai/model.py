"""
Neural network model for circuit accuracy prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CircuitPredictor(nn.Module):
    """4-layer neural network for circuit accuracy prediction"""
    
    def __init__(self, input_dim=903, dropout_rate=0.2):
        super().__init__()
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal for ReLU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through network"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = torch.sigmoid(self.fc4(x))
        return x
    
    def predict(self, x):
        """Prediction without gradient computation"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
    def save(self, path):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.fc1.in_features,
            'dropout_rate': self.dropout.p
        }, path)
        
    @classmethod
    def load(cls, path, device='cuda'):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    model = CircuitPredictor(input_dim=303)  # Our data has 303 features
    print(f"Model architecture: {model}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    test_input = torch.randn(32, 303)  # Batch of 32 circuits
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
