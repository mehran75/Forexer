import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.DoubleTensor')


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers, target_length, device):
        super(SimpleGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size * seq_length, target_length)

    def forward(self, x):
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = out.reshape(x.shape[1], -1)
        out = self.fc(out)

        return out

    def train_model(self, num_epochs, batch_size, criterion, optimizer, data):
        X_train = data[0]
        y_train = data[1]

        for epoch in range(num_epochs):
            counter = 0
            for i in range(batch_size, X_train.shape[0], batch_size):
                counter = i
                optimizer.zero_grad()

                scores = self.forward(X_train[i - batch_size:i].view(self.seq_length, batch_size, -1))
                loss = criterion(scores, y_train[i - batch_size:i])

                loss.backward()

                optimizer.step()

                print('', end='\r epoch: {:3}/{:3}, loss: {:10.8f}, completed: {:.2f}%'.format(epoch + 1,
                                                                                               num_epochs,
                                                                                               loss.item(),
                                                                                               (i / X_train.shape[
                                                                                                   0]) * 100))
            if counter < X_train.shape[0]:
                optimizer.zero_grad()

                bs = X_train.shape[0] - counter

                scores = self.forward(X_train[counter:].view(self.seq_length, bs, -1))
                loss = criterion(scores, y_train[counter:])

                loss.backward()

                optimizer.step()

                print('', end='\r epoch: {:3}/{:3}, loss: {:10.8f}, completed: {:.2f}%'.format(epoch + 1,
                                                                                               num_epochs,
                                                                                               loss.item(), 100))

            print('\n')

    def predict(self, x):

        self.eval()
        y_preds = self(x.view(self.seq_length, x.shape[0], -1))
        self.train()

        return y_preds
