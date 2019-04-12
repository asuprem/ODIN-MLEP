import codecs
import datetime
import io
import json
import numpy as np
import sqlite3
import sys
import time

def std_flush(*args, **kwargs):
    """Write a space-separated tuple of arguments to the standard output and flush its buffers."""
    print(" ".join(map(str, args)))
    sys.stdout.flush()

def readable_time(format="%M:%S"):
    """Return a string representing the current time, controlled by an explicit format string.

    format -- [str] Format string.
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime(format)

def ms_to_readable(ms):
    return datetime.datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%d %H:%M:%S')

def time_to_id(ms=None, lval = 5):
    DICTA={str(idx):item for idx,item in enumerate("abcdefghij")}
    if ms is None:
        ms = time.time()
    ms_str = ("%."+str(lval)+"f")%ms
    ms_str = ms_str[:-(lval+1)]+ms_str[-lval:]
    return ''.join([DICTA[item] for item in ms_str])

def adapt_array(arr):
    # [RODRIGO]: Pushing numpy array to sqlite (conversion).
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    # [RODRIGO]: Pushing numpy array to sqlite (conversion).
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def load_json(json_):
    return json.load(codecs.open(json_, encoding='utf-8'))
