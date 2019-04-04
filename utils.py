import sys, pdb
from datetime import datetime
import time

def std_flush(*args,**kwargs):
    print(" ".join(map(str,args)))
    sys.stdout.flush()

def readable_time():
    #return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return datetime.fromtimestamp(time.time()).strftime('%M:%S')

def ms_to_readable(ms):
    return datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%d %H:%M:%S')