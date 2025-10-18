"""
Registry of available network architectures.
"""
from .gru import GRU
from .simple import SimpleRNN

Networks = {
    'gru': GRU,
    'simple': SimpleRNN,
}
