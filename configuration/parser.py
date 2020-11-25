import yaml


class Configuration:
    def __init__(self, config):
        self.plot = config['plot']
        self.delay = config['delay']
        self.mode = config['mode']

        self.model = Model(config['model'])
        self.data = Data(config['data'])


class Model:
    def __init__(self, config):
        self.type = config['type']
        self.pre_trained = config['pretrained-weights']
        self.save_path = config['save-path']

        self.parameters = Parameter(config['parameters'])


class Parameter:
    def __init__(self, config):
        self.label = config['label']
        self.time_window = config['time-window']
        self.sequence_length = config['sequence-length']
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

