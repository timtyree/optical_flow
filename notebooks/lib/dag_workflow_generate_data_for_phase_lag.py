#TODO: fill in design pattern.

def load(filename):
  pass
  '''TODOD: simple gui viewer for selecting files '''
    ...

def clean(data):
  '''measure intensity time series for a time interval sample from tiffstack.  write subroutines using import jax.'''
    ...

def analyze(sequence_of_data):
  '''compute cross correlation functions of time series values along with measures of causality that could be scale invarient.'''
    ...

def store(result, print_file_name, initializeQ = False):
  '''write results to a print_file_name.'''
    with open(..., 'w') as f:
        f.write(result)

dsk = {'load-1': (load, 'myfile.a.data'),
       'load-2': (load, 'myfile.b.data'),
       'load-3': (load, 'myfile.c.data'),
       'clean-1': (clean, 'load-1'),
       'clean-2': (clean, 'load-2'),
       'clean-3': (clean, 'load-3'),
       'analyze': (analyze, ['clean-%d' % i for i in [1, 2, 3]]),
       'store': (store, 'analyze')}
#TODO: what's ^this?

from dask.multiprocessing import get
get(dsk, 'store')  # executes in parallel