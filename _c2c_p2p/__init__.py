# -*- coding: utf-8 -*-
import logging
from logging import handlers
import os
import sys
import warnings

from .const import LOG_LOC, OUTPUT_LOC, SCHEMA_CONFIG_LOC

warnings.filterwarnings("ignore")
if not os.path.exists(LOG_LOC):
    os.makedirs(LOG_LOC)
if not os.path.exists(f'{OUTPUT_LOC}/trees'):
    os.makedirs(f'{OUTPUT_LOC}/trees')
if not os.path.exists(f'{OUTPUT_LOC}/reports'):
    os.makedirs(f'{OUTPUT_LOC}/reports')
if not os.path.exists(f'{OUTPUT_LOC}/_temp_file'):
    os.makedirs(f'{OUTPUT_LOC}/_temp_file')
file_hdlr = handlers.TimedRotatingFileHandler(
    filename=f'{LOG_LOC}/.log', when='D', backupCount=7, encoding='utf-8')
fmt = '%(asctime)s.%(msecs)03d - %(levelname)s - %(filename)s - line %(lineno)d: %(message)s'
info_hdlr = logging.StreamHandler(sys.stdout)
info_hdlr.setLevel(logging.INFO)
file_hdlr.setLevel(logging.INFO)
logging.basicConfig(level=0, format=fmt, handlers=[
                    file_hdlr, info_hdlr], datefmt='%Y-%m-%d %H:%M:%S')

from .const import (SPLITER, PK, PK2, Task, dataGeneratorType, dataFuncType,
                    dataRangeType, configType, AlgorithmCodes, TableNames, Dtypes, 
                    ValueColumns, SchemaTableRefs, DataCateNames, FieldInfoNames)
from .utils import BufferList, ExtendedColumn, ColumnManager, SetCode, SetCodeManager

__all__ = [OUTPUT_LOC, LOG_LOC, SPLITER, PK, PK2, Task, dataRangeType, dataFuncType, 
           dataGeneratorType, configType, ValueColumns, TableNames, Dtypes, 
           AlgorithmCodes, SchemaTableRefs, DataCateNames, FieldInfoNames, 
           BufferList, SetCode, SetCodeManager, ExtendedColumn, ColumnManager]

from ._data import DataSet

__all__ += [DataSet]

if not os.path.exists(SCHEMA_CONFIG_LOC):
    from ._config import *
