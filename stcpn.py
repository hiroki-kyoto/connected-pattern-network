# STCPN : Spatio-Temporally Connected Pattern Network

import numpy as np

import utils


class STCPN(object):
    '''
    STCPN : Spatio-Temporally Connected Pattern Network.
    Filters(neurons) are connected to each other by
    oscillations. Input signals contain both temporal
    and temporal information, such as Images and Texts.
    '''
    def __init__(self, conf):
        print 'init'

        conf = utils.handle_conf_json(conf)
        print 'construct network with configuration'

        assert 'x' in conf
        assert 'y' in conf
        assert 'hidden' in conf

        self.x_dim = conf['x']
        self.y_dim = conf['y']
        self.hidden_dim = conf['hidden']
        self.learning_rate = 0.01
        self.decay_rate = 0.01

        if 'learning_rate' in conf:
            self.learning_rate = conf['learning_rate']
        elif 'decay_rate' in conf:
            self.decay_rate = conf['decay_rate']

        # build filters layer by layer


    def __enter__(self):
        print 'enter'

    def __exit__(self, exc_type, exc_val, exc_tb):
        print 'exit'

    # define addition operator for network
    def __add__(self, other):
        assert type(other) == STCPN
        # Create a new network combining all
        # filters of two networks, and add new
        # connections between each other.
        print 'network merging'

    def __and__(self, other):
        assert type(other) == STCPN
        # If current network do respond to
        # input, then deliver it to next
        # Else terminated with False.
        # A handler for input is returned
        print 'network A && network B'

    def __or__(self, other):
        assert type(other) == STCPN
        # If current network do not respond to
        # input, then deliver it to next
        # Else terminated with True.
        # A handler for input is returned
        print 'network A || network B'

    def __call__(self, *args, **kwargs):
        # reset the initial settings of this object
        if len(args)>0:
            conf_json = args[0]
        if 'conf' in kwargs:
            conf_json = kwargs.pop('conf')
            assert len(kwargs)>0
        conf = utils.handle_conf_json(conf_json)
        print conf

    def train(self, x, y):
        s = [x, y]