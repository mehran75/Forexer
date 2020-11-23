import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.set_default_tensor_type('torch.DoubleTensor')

input_size = 1
seq_length = 9
num_layers = 2
hidden_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device='cpu'

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size * seq_length, 1)
        
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, input_size, self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(input_size, -1)
        out = self.fc(out)
        
        return out
    
    
    def predict(self, X):
        self.eval()
        y_preds = torch.zeros(X.shape[0], 1).to(device)

        for i in range(X.shape[0]):
            y_preds[i] = self(X[i].view(seq_length, 1, -1))

        self.train()

        return y_preds
        
    def predict_stream(self, X):
        pass