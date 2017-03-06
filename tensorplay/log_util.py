import os
import sys
import logging.config
from contextlib import contextmanager


def configure_logger():
    default_config = {
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(asctime)s %(levelname)s: %(message)s'
            }
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'default',
                'class': 'logging.StreamHandler',
            }
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'WARN',
            },
            'tensorplay': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(default_config)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
