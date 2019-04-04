import sys, pdb
from datetime import datetime
import time
import sqlite3
import numpy as np
import io


def std_flush(*args,**kwargs):
    print(" ".join(map(str,args)))
    sys.stdout.flush()

def readable_time():
    #return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return datetime.fromtimestamp(time.time()).strftime('%M:%S')

def ms_to_readable(ms):
    return datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%d %H:%M:%S')

def time_to_id(ms=None):
    DICTA={str(idx):item for idx,item in enumerate("abcdefghij")}
    if ms is None:
        ms = time.time()
    ms_str = "%.2f"%time.time()
    ms_str = ms_str[:-3]+ms_str[-2:]
    return ''.join([DICTA[item] for item in ms_str])


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)