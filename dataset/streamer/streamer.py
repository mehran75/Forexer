import requests
from datetime import datetime, timezone
import json
import numpy as np


class Streamer:
    """

    """

    def __init__(self, **kwargs):

        self.url = 'https://www.alphavantage.co/query'

        self.api_key = kwargs.get('apikey', 'demo')

        # optionals
        self.file_path = kwargs.get('file_path', '')
        self.function = kwargs.get('function', 'FX_INTRADAY')
        self.interval = kwargs.get('interval', '1M')
        self.from_symbol = kwargs.get('from_symbol', 'EUR')
        self.to_symbol = kwargs.get('to_symbol', 'USD')

    def save_file(self, data, file_path):
        pass

    def retrieve(self, label, shape):
        """
        send request to retrieve information
        Note: call authenticate before calling this
        :param label: label for training/predicting
        :param shape: required shape. e.g (batch_size (not supported), input_size, sequence_length)
        :return: numpy array of requested sequence
        """

        params = {
            'function': self.function,
            'from_symbol': self.from_symbol,
            'to_symbol': self.to_symbol,
            'interval': self.interval,
            'apikey': self.api_key
        }
        res = requests.get(self.url, params=params)

        if res.status_code != 200:
            raise Exception("Could not receive correct data. response code: {}".format(res.status_code))

        try:
            j = json.loads(res.text.strip())
            j = j[list(j.keys())[1]]

            data = list(j.values())[0:shape[2]]
            key = ''
            for k in data[0].keys():
                if label.lower() in k.lower():
                    key = k
                    break

            data = [d[key] for d in data]
        except Exception as e:
            print(res.text)
            raise e
        return np.array(data, dtype=np.float64).reshape(shape)
