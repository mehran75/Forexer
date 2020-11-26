import time

import requests
from threading import Thread
from datetime import datetime, timezone


class Streamer(Thread):
    """

    """
    SYMBOLS_ALL = ['EUR/USD', 'EUR/CAD', 'USD/JPY', 'GBP/USD', 'EUR/GBP', 'USD/CHF', 'AUD/NZD',
                   'CAD/CHF', 'CHF/JPY', 'EUR/AUD', 'EUR/JPY', 'EUR/CHF', 'USD/CAD', 'EUR/NZD',
                   'AUD/USD', 'GBP/JPY', 'AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'EUR/NOK', 'USD/SEK',
                   'GBP/CAD', 'GBP/CHF', 'NZD/JPY', 'NZD/USD', 'USD/NOK']

    def __init__(self, data, sequence_length, **kwargs):
        super(Streamer, self).__init__()

        self.data = data  # shared variable

        self.sequence_length = sequence_length
        self.url = 'https://webrates.truefx.com/rates/connect.html'
        self.id = ''

        self.stop_running = False
        self.session = requests.session()

        # optionals
        self.delay = kwargs.get('delay', 1)
        self.file_path = kwargs.get('file_path', '')
        self.username = kwargs.get('user', ['', ''])[0]
        self.password = kwargs.get('user', ['', ''])[1]
        self.currencies = kwargs.get('currency', [self.SYMBOLS_ALL[0]])

    def authenticate(self):

        print('Authenticating user: {}'.format(self.username))
        res = self.session.get(self.url, params={
            'u': self.username,
            'p': self.password,
            'q': 'defualt',
            'f': 'csv',
            'c': ','.join(self.currencies),
            's': 'n'
        })
        if res.status_code != 200:
            raise Exception('Failed to authenticate user. code: {}'.format(res.status_code))

        self.id = res.text.strip()

    def end_session(self):
        self.session.get(self.url, params={
            'di': self.id
        })

    def save_file(self, data, file_path):
        pass

    def run(self):
        self.authenticate()

        print('Streaming has started...')
        while not self.stop_running:  # todo: change the condition with ctrl+c command

            res = self.session.get(self.url, params={'id': self.id})

            if res.status_code == 200:
                res = res.text.strip()
                if res != '' and res not in self.data:
                    self.data.append(res)

            time.sleep(self.delay)

    def join(self):
        self.stop_running = True
        self.end_session()
        super().join()
