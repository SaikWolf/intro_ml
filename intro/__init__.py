

__version__ = '0.0.1'

import os
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if 'lib/python3' in _pkg_dir:
    _dev_dir = os.path.join(os.environ['HOME'],'.local','src','intro_dev')
    if not os.path.isdir(_dev_dir):
        os.makedirs(_dev_dir,exist_ok=True)
else:
    _dev_dir = os.path.dirname(_pkg_dir)
_dta_dir = os.path.join(_dev_dir,'datasets')

def get_dataset_dir():
    return _dta_dir
def get_dev_dir():
    return _dev_dir


del os
from . import models
from . import loggers