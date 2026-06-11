"""
Utility functions for PyTorch recurrent training.
"""
import datetime
import errno
import io
import os
import pickle
import signal
import sys
from collections import OrderedDict


def println(line):
    """Print line without newline and flush."""
    sys.stdout.write(line)
    sys.stdout.flush()


def mkdir_p(path):
    """
    Portable mkdir -p
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def print_dict(settings_, indent=2, title=None):
    """
    Pretty print.
    """
    if isinstance(settings_, (list, tuple)):
        settings = OrderedDict()
        for s in settings_:
            settings.update(s)
    else:
        settings = settings_

    if title is not None:
        print(title)
    maxlen = max([len(s) for s in settings])
    for k, v in settings.items():
        print(indent*' ' + '{}:{}{}'.format(k, (maxlen - len(k) + 1)*' ', v))
    sys.stdout.flush()


#=========================================================================================
# Paths
#=========================================================================================

def get_here(file):
    """Get the directory containing the given file."""
    return os.path.abspath(os.path.dirname(file))


#=========================================================================================
# Pickle
#=========================================================================================

def save(filename, obj):
    """
    Save object to pickle file.
    Disable keyboard interrupt while pickling.
    """
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    signal.signal(signal.SIGINT, s)


def load(filename, map_location='cpu'):
    """Load object from pickle file, mapping torch tensor storage to CPU by default."""
    with open(filename, 'rb') as f:
        try:
            import torch
        except ImportError:
            return pickle.load(f)

        load_from_bytes = getattr(torch.storage, '_load_from_bytes', None)
        if load_from_bytes is None or map_location is None:
            return pickle.load(f)

        def load_from_bytes_mapped(storage_bytes):
            return torch.load(
                io.BytesIO(storage_bytes),
                map_location=map_location,
                weights_only=False
            )

        torch.storage._load_from_bytes = load_from_bytes_mapped
        try:
            return pickle.load(f)
        finally:
            torch.storage._load_from_bytes = load_from_bytes


#=========================================================================================
# Reporting
#=========================================================================================

def elapsed_time(tstart):
    """Get elapsed time as formatted string."""
    tnow = datetime.datetime.now()
    totalsecs = (tnow - tstart).total_seconds()

    hrs = int(totalsecs//3600)
    mins = int(totalsecs%3600)//60
    secs = int(totalsecs%60)

    return '{}h {}m {}s elapsed'.format(hrs, mins, secs)
