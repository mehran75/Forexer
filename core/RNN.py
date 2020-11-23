import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.set_default_tensor_type('torch.DoubleTensor')


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers, device):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.seq_length = seq_length
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size * seq_length, 1)
        
    def forward(self, x):
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers, self.input_size, self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(self.input_size, -1)
        out = self.fc(out)
        
        return out
    
    
    def predict(self, X):
        self.eval()
        y_preds = torch.zeros(X.shape[0], 1).to(self.device)

        for i in range(X.shape[0]):
            y_preds[i] = self(X[i].view(self.seq_length, 1, -1))

        self.train()

        return y_preds
        
    def predict_stream(self, X):
        pass