import os
import sys
import time
import torch
import PIL
import logging
import warnings
from functools import wraps


def show_runtime(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("[Time] %s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


def get_machine_info():
    info = "\n >> Python  Version  : {:}".format(
        sys.version.replace('\n', ' '))
    info += "\n >> Pillow  Version  : {:}".format(PIL.__version__)
    info += "\n >> PyTorch Version  : {:}".format(torch.__version__)
    info += "\n >> cuDNN   Version  : {:}".format(
        torch.backends.cudnn.version())
    info += "\n >> CUDA available   : {:}".format(torch.cuda.is_available())
    info += "\n >> CUDA GPU numbers : {:}".format(torch.cuda.device_count())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        info += "\n >> CUDA_VISIBLE_DEVICES={:}".format(
            os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        info += "\n >> Does not set CUDA_VISIBLE_DEVICES"
    return info


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    print(__file__)
    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s : [%(levelname)s] :: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


def filter_warnings():
    warnings.filterwarnings(
        "ignore", message=torch.optim.lr_scheduler.SAVE_STATE_WARNING)
