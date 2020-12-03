import yaml


class Configuration:
    def __init__(self, config):
        self.plot = config['plot']
        self.mode = config['mode']
        self.stream_train = config['stream-train']

        self.currency = Currency(config['currency'])
        self.model = Model(config['model'])
        self.data = Data(config['data'])


class Currency:
    def __init__(self, config):
        self.from_symbol = config['from_symbol']
        self.to_symbol = config['to_symbol']
        self.function = config['function']
        self.interval = config['interval']

    def interval_to_seconds(self):
        key_map = {
            '1min': 1 * 60,
            '5min': 5 * 60,
            '15min': 15 * 60,
            '30min': 30 * 60,
            '60min': 60 * 60,
        }
        if self.function == 'FX_INTRADAY':
            return key_map[self.interval]
        elif self.function == 'FX_DAILY':
            return 3600 * 24
        elif self.function == 'FX_WEEKLY':
            return 3600 * 24 * 7
        else:
            Exception("Not Implemented")


class Model:
    def __init__(self, config):
        self.type = config['type']
        self.pre_trained = config['pretrained-weights']
        self.save_path = config['save-path']

        self.parameters = Parameter(config['parameters'])


class Parameter:
    def __init__(self, config):
        self.label = config['label']
        self.sequence_length = config['time-window']
        self.target_length = config['preceding-window']
        self.input_size = config['input-size']
        self.num_layers = config['num-layers']
        self.hidden_szie = config['hidden-size']
        self.device = config['device']
        self.lr = config['lr']
        self.num_epochs = config['num-epochs']
        self.batch_size = config['batch-size']


class Data:
    def __init__(self, config):
        self.train_path = config['train-path']
        self.test_path = config['test-path']
        self.dev_size = config['dev-size']


def load_configuration(config_path="configuration/parameters.yml"):
    with open(config_path, 'r') as stream:
        parameters = yaml.safe_load(stream)

    return Configuration(parameters)
