import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from configuration.parser import load_configuration
from core.RNN import SimpleRNN
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import os

from dataset.streamer.streamer import Streamer


def create_sequence(data, label, time_window):
    size = int(data.shape[0] / time_window) + data.size % time_window

    X = np.zeros((size, time_window - 1))
    y = np.zeros((size, 1))

    counter = 0
    for i in range(time_window, data.shape[0], time_window):
        X[counter] = (data[label].values[i - time_window: i - 1])
        y[counter] = (data[label].values[i])
        counter += 1

    return X, y



def to_tensor(a, device):
    return torch.from_numpy(a).to(device)


def extract_data(data, shape, start_index):
    """

    :param data: array of raw data
    :param shape: required shape. e.g (input_size, sequence_length)
    :param start_index: data[start_index:start_index+sequence_length]
    :return: numpy array for
    """

    if len(data) == 0:
        return None

    seq = data[shape[0]: start_index + shape[1]+1]

    seq = [''.join(i.split(',')[2:4]) for i in seq]
    print(seq)
    return np.array(seq, dtype=np.float).reshape(shape)


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
    time_window = config.model.parameters.time_window

    input_size = config.model.parameters.input_size
    seq_length = config.model.parameters.sequence_length
    num_layers = config.model.parameters.num_layers
    hidden_size = config.model.parameters.hidden_szie
    batch_size = config.model.parameters.batch_size
    learning_rate = config.model.parameters.lr
    num_epochs = config.model.parameters.num_epochs
    device = torch.device('cuda' if torch.cuda.is_available() and config.model.parameters.device == 'cuda' else 'cpu')

    # creating model
    model = SimpleRNN(input_size, hidden_size, seq_length, num_layers, device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loading pre-trained weights
    if config.model.pre_trained != '':
        print('Loading pre-trained model')
        try:
            model_checkpoints = torch.load(config.model.pre_trained, map_location=device)
            model.load_state_dict(model_checkpoints['model_state_dict'])
            optimizer.load_state_dict(model_checkpoints['optimizer_state_dict'])
            criterion.load_state_dict(model_checkpoints['criterion_state_dict'])
        except KeyboardInterrupt as e:
            exit(0)
        except Exception as e:
            print('failed to load the model properly. Error: {}'.format(e))

    if config.mode == 'train':
        data = pd.read_csv(config.data.train_path)
        print('Loaded {} data with shape of {}'.format(config.data.train_path, data.shape))

        # to sequence
        X, y = create_sequence(data, label, time_window)

        # split to train and dev set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.data.dev_size)

        # train model
        print('Training the model...')
        model.train_model(num_epochs, batch_size, criterion, optimizer,
                          (to_tensor(X_train, device), to_tensor(y_train, device)))

        # evaluating model
        print('Evaluating the model...')
        y_preds = model.predict(to_tensor(X_test, device))
        y_preds = y_preds.cpu().detach().numpy()
        score = r2_score(y_preds, y_test)

        if config.model.save_path != '':
            now = datetime.now()
            print('Saving model. path: {}'.format(config.model.save_path +
                                                  str(now.date()) +
                                                  '--' +
                                                  str(now.time())[:5].replace(':', '-') +
                                                  '-{:3f}'.format(score) + '.zip'
                                                  ))

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict()
            },
                config.model.save_path +
                str(now.date()) +
                '--' +
                str(now.time())[:5].replace(':', '-') +
                '-{:3f}'.format(score) + '.zip')

            print('R2 score: {:.3f}'.format(score))

    elif config.mode == 'test':
        data = pd.read_csv(config.data.test_path)
        print('Loaded {} data with shape of {}'.format(config.data.train_path, data.shape))

        # to sequence
        X_test, y_test = create_sequence(data, label, time_window)

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
        X = np.zeros((0, seq_length))
        y_prev = np.array([])
        y = np.array([])

        streamer = Streamer(data, seq_length,
                            delay=config.delay,
                            user=(os.environ.get('USERNAME'), os.environ.get('PASSWORD')))

        try:
            start_index = 0
            streamer.start()
            time.sleep(10)
            while len(data) <= seq_length:
                eta = (config.delay / 60) * (seq_length - (len(data)))
                print('', end='\r preparing the first sequence. ETA: {}'.format(
                    '{:.2f} min'.format(eta) if eta > 1 else '{} sec'.format(int(eta * 60))))
                time.sleep(config.delay)

            print('\n')
            while True:
                plt.close()
                X = np.append(X, extract_data(data, (input_size, seq_length), start_index))
                print(X)
                pred = model.predict(torch.from_numpy(X).to(device))

                y = np.append(y, pred.cpu().detach().numpy())
                y_prev = np.append(y_prev, X[-1])

                if config.plot:
                    plt.plot(y, label='predict')
                    plt.plot(y_prev, label='truth')
                    plt.legend()
                    plt.show()

                start_index += 1
                time.sleep(config.delay + 1)
        except KeyboardInterrupt:
            print('\nCtrl-c detected, stopping the streamer')
            streamer.join()
            exit(0)
        except Exception as e:
            print('\nAn error occurred, Error: {}'.format(e))
            streamer.join()
            exit(1)
