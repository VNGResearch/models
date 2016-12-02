
###############################################################################
# Model application
###############################################################################

import os
path = os.path.dirname(__file__)

import sys
import network
from vec4net import make_vec
import numpy as np

# Replace this with your model's result
JSON = os.path.join(path, 'params.net')
net = network.load(JSON)


def _get_iob(arr):
    d = {0: 'i', 1: 'o', 2: 'b'}
    n = np.argmax(arr)
    return d[n]

def _classify(token_list):
    "Classify a list of token"
    result = []
    sen_vec = make_vec(token_list)
    for x in sen_vec:
        result.append(_get_iob(net.feedforward(x)))
    return result

def _make_words(token_list, iob_list):
    "Make segmented words from token list and corresponding iob list"
    if not iob_list: return
    t = token_list[0:1]
    tokens = []
    for i in range(1, len(iob_list)):
        if iob_list[i] == 'i':
            t.append(token_list[i])
            continue
        if iob_list[i] == 'b':
            if not t:
                t = token_list[i:i+1]
                tokens.append(t)
                t = []
            else:
                tokens.append(t)
                t = token_list[i:i+1]
            continue
        if iob_list[i] == 'o':
            if t:
                tokens.append(t)
                t = []
            tokens.append(token_list[i:i+1])
    if t: tokens.append(t)
    return ['_'.join(tok) for tok in tokens]

def tokenize(txt):
    words = txt.split()
    token_list = txt.lower().split()
    iob_list = _classify(token_list)
    return _make_words(words, iob_list)
