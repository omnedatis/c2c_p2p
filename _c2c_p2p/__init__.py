# -*- coding: utf-8 -*-
import logging
from logging import handlers
import os
import sys
import warnings

from .common import LOG_LOC, OUTPUT_LOC

warnings.filterwarnings("ignore")
if not os.path.exists(LOG_LOC):
    os.makedirs(LOG_LOC)
if not os.path.exists(f'{OUTPUT_LOC}/trees'):
    os.makedirs(f'{OUTPUT_LOC}/trees')
if not os.path.exists(f'{OUTPUT_LOC}/reports'):
    os.makedirs(f'{OUTPUT_LOC}/reports')
file_hdlr = handlers.TimedRotatingFileHandler(
    filename=f'{LOG_LOC}/.log', when='D', backupCount=7, encoding='utf-8')
fmt = '%(asctime)s.%(msecs)03d - %(levelname)s - %(filename)s - line %(lineno)d: %(message)s'
info_hdlr = logging.StreamHandler(sys.stdout)
info_hdlr.setLevel(logging.INFO)
file_hdlr.setLevel(logging.INFO)
logging.basicConfig(level=0, format=fmt, handlers=[
                    file_hdlr, info_hdlr], datefmt='%Y-%m-%d %H:%M:%S')

from .common import (SPLITER, Task, dataGeneratorType,
                     ExtendedColumn, ColumnManager, AlgorithmCodes)
from ._loader import DataSet

if not os.path.exists('_c2c_p2p/_local_db_config.json'):
    from .gen_config import *

__all__ = [OUTPUT_LOC, LOG_LOC, SPLITER, Task,
           dataGeneratorType, ExtendedColumn, DataSet,
           ColumnManager, AlgorithmCodes]


