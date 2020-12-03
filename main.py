import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from configuration.parser import load_configuration
from core.GRU import SimpleGRU
from core.LSTM import SimpleLSTM
from core.RNN import SimpleRNN
from core.RMSLE import RMSLELoss
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import os

from dataset.streamer.streamer import Streamer


def create_sequence(data, label, lengths):
    """
    create sequence from time series dataset
    :param data: pandas dataframe
    :param label: target label
    :param lengths: tuple of (sequence_length, target_length)
    :return:
    """

    seq_length = lengths[0]
    target_length = lengths[1]

    inputs = np.zeros((data.shape[0] - target_length, seq_length))
    targets = np.zeros((data.shape[0] - target_length, target_length))

    for index, i in enumerate(range(seq_length, data.shape[0] - target_length)):
        inputs[index] = (data[label].values[i - seq_length: i])
        targets[index] = (data[label].values[i: i + target_length])

    return inputs, targets


def to_tensor(a, device):
    """
    numpy array to PyTorch tensor
    :param a: numpy array
    :param device: 'cuda' or 'cpu'
    :return: PyTorch tensor
    """
    return torch.from_numpy(a).to(device)


def prepare_model(type, input_size, hidden_size, seq_length, num_layers, device):
    """
    create and load the specified model

    :param type: "RNN", "LSTM", or "GRU"
    :param input_size: model parameters
    :param hidden_size: model parameters
    :param seq_length: model parameters
    :param num_layers: model parameters
    :param device: 'cuda' or 'cpu'
    :return: model, criterion(loss function), optimizer
    """

    if type == 'RNN':
        model = SimpleRNN(input_size, hidden_size, seq_length, num_layers, target_length, device).to(device)
    elif type == 'GRU':
        model = SimpleGRU(input_size, hidden_size, seq_length, num_layers, target_length, device).to(device)
    elif type == 'LSTM':
        model = SimpleLSTM(input_size, hidden_size, seq_length, num_layers, target_length, device).to(device)
    else:
        raise NotImplemented("type {} doesn't exist".format(type))

    criterion = RMSLELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loading pre-trained weights
    if config.model.pre_trained != '':
        print('Loading pre-trained model')
        try:
            model_checkpoints = torch.load(config.model.pre_trained, map_location=device)
            model.load_state_dict(model_checkpoints['model_state_dict'])
            optimizer.load_state_dict(model_checkpoints['optimizer_state_dict'])
            criterion.load_state_dict(model_checkpoints['criterion_state_dict'])
        except Exception as e:
            print('failed to load the model properly. Error: {}'.format(e))

    return model, criterion, optimizer


if __name__ == '__main__':

    print('\n********************************************************')
    print('                         Forexer')
    print('********************************************************\n\n')

    print('GPU is {}available'.format('' if torch.cuda.is_available() else 'not '))
    print('GPU device count: {}'.format(torch.cuda.device_count()))
    torch.set_default_tensor_type('torch.DoubleTensor')

    # load configurations
    config_path = sys.argv[1]
    print(config_path)
    if config_path != '':
        config = load_configuration(config_path)
    else:
        config = load_configuration()

    # mapping parameters
    label = config.model.parameters.label

    input_size = config.model.parameters.input_size
    seq_length = config.model.parameters.sequence_length
    target_length = config.model.parameters.target_length
    num_layers = config.model.parameters.num_layers
    hidden_size = config.model.parameters.hidden_szie
    batch_size = config.model.parameters.batch_size
    learning_rate = config.model.parameters.lr
    num_epochs = config.model.parameters.num_epochs
    device = torch.device('cuda' if torch.cuda.is_available() and config.model.parameters.device == 'cuda' else 'cpu')

    # creating model

    model, loss, optimizer = prepare_model(config.model.type,
                                           input_size, hidden_size, seq_length, num_layers, device)

    if config.mode == 'train':
        data = pd.read_csv(config.data.train_path)
        print('Loaded {} data with shape of {}'.format(config.data.train_path, data.shape))

        # to sequence
        X, y = create_sequence(data, label, (seq_length, target_length))

        # split to train and dev set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.data.dev_size)

        # train model
        print('Training the model...')
        model.train_model(num_epochs, batch_size, loss, optimizer,
                          (to_tensor(X_train, device), to_tensor(y_train, device)))

        if config.model.save_path != '':
            now = datetime.now()
            print('Saving model. path: {}'.format(config.model.save_path +
                                                  config.model.type + '-' +
                                                  str(now.date()) +
                                                  '--' +
                                                  str(now.time())[:5].replace(':', '-') + '.zip'
                                                  ))

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': loss.state_dict()
            },
                config.model.save_path +
                config.model.type + '-' +
                str(now.date()) +
                '--' +
                str(now.time())[:5].replace(':', '-') + '.zip')

        # evaluating model
        print('Evaluating the model...')
        y_preds = model.predict(to_tensor(X_test, device))
        y_preds = y_preds.cpu().detach().numpy()
        score = r2_score(y_preds, y_test)

        print('R2 score: {:.3f}'.format(score))

    elif config.mode == 'test':
        data = pd.read_csv(config.data.test_path)
        print('Loaded {} data with shape of {}'.format(config.data.train_path, data.shape))

        # to sequence
        X_test, y_test = create_sequence(data, label, (seq_length, target_length))

        # evaluating model
        print('Evaluating the model...')
        y_preds = model.predict(to_tensor(X_test, device))

        y_preds = y_preds.cpu().detach().numpy()
        score = r2_score(y_preds, y_test)
        print('R2 score: {:.3f}'.format(score))

        if config.plot:
            plt.plot(y_preds[:-2], label='predict')
            plt.plot(y_test[:-2], label='truth')
            plt.legend()
            plt.show(block=True)

    elif config.mode == 'stream':

        data = []
        X = np.zeros((0, input_size, seq_length))
        Y = np.zeros((0, 1))
        y_prev = np.zeros((0, 1))

        streamer = Streamer(apikey=os.environ.get('apikey', 'demo'),
                            interval=config.currency.interval,
                            from_symbol=config.currency.from_symbol,
                            to_symbol=config.currency.to_symbol,
                            function=config.currency.function)

        try:
            start_index = 0
            while True:

                x = streamer.retrieve(label, (1, input_size, seq_length))
                pred = model.predict(torch.from_numpy(x).to(device))
                pred = pred.cpu().detach().numpy().flatten()

                X = np.append(X, x, axis=0)

                if start_index > 0:
                    Y = np.append(Y, pred[-1])
                    y_prev = np.append(y_prev, X[-1, 0, -2].reshape(1, -1), axis=0)
                    print('Predicted {:.5f}, truth: {}'.format(Y[-target_length], y_prev[-1]))

                    if config.stream_train:
                        model.train_model(1, 1, loss, optimizer,
                                          (torch.from_numpy(x).to(device), torch.from_numpy(y_prev[-1]).to(device)))
                else:
                    Y = np.append(Y, pred)
                    print('Predicted: {:.5f}'.format(Y[-target_length]))

                feed = streamer.retrieve(label, (1, input_size, seq_length))
                if type(feed) == int:
                    raise Exception("Failed to receive new feed")
                data.append(feed)

                if config.plot:
                    plt.clf()
                    plt.plot(Y[:5 + y_prev.shape[0]], label='predict')
                    if start_index > 0:
                        plt.plot(y_prev, label='truth')
                    plt.legend()
                    plt.pause(config.currency.interval_to_seconds())
                start_index += 1

        except KeyboardInterrupt:
            print('\nCtrl-c detected, stopping the streamer')
            exit(0)
        except Exception as e:
            raise e
