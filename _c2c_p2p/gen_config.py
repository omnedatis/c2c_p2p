# -*- coding: utf-8 -*-
from collections import defaultdict
from email.policy import default
import logging
import json
import traceback

import numpy as np
import pandas as pd

from .common import (SCHEMA_CONFIG_LOC, DataCateNames, FieldInfoNames,
                     SchemaTableRefs, ValueColumns, AlgorithmCodes, configType)
try:
    configs: configType = {}
    for each_t in SchemaTableRefs:
        table = {
            "table_name": each_t.target_file,
            "table_fields": {},
        }
        field_info: pd.DataFrame = pd.read_excel(
            each_t.file_name, each_t.main_def).set_index(FieldInfoNames.CODE.value)

        set_sheet:np.ndarray = pd.read_excel(
            each_t.file_name, each_t.set_def, header=None).values
        set_elements = field_info.loc[field_info[FieldInfoNames.CATE.value]
                                      == DataCateNames.SET.value][FieldInfoNames.CATE_CODE.value].tolist()
        set_defs = {}
        for rid, row in enumerate(set_sheet):
            for cid, cell in enumerate(row.tolist()):
                if cell in set_elements:
                    set_defs[cell] = {
                        i:int(j) for i, j in set_sheet[rid+1:rid+3].T.tolist() if j == j}

        value_sheet = pd.read_excel(each_t.file_name, each_t.value_def)
        column_names = {i: col for i, col in enumerate(value_sheet)}
        value_defs = defaultdict(lambda: dict({'dtype':'', 'range':{}}))
        for rid, row in enumerate(value_sheet.values):
            for cid, cell in enumerate(row):
                cname = column_names[cid]

                if cname == ValueColumns.NAME.value:
                    name = cell
                    if name is None:
                        break
                elif cname == ValueColumns.VALUEDTYPE.value:
                    value_defs[name]['dtype'] += cell.lower()

                elif cname == ValueColumns.LC.value:
                    value_defs[name]['range'].update({
                        ValueColumns.LC.value: cell if (cell == cell) else None
                    })
                elif cname == ValueColumns.RC.value:
                    value_defs[name]['range'].update({
                        ValueColumns.RC.value: cell if (cell == cell) else None
                    })
                elif cname == ValueColumns.RO.value:
                    value_defs[name]['range'].update({
                        ValueColumns.RO.value: cell if (cell == cell) else None
                    })
                elif cname == ValueColumns.LO.value:
                    value_defs[name]['range'].update({
                        ValueColumns.LO.value: cell if (cell == cell) else None
                    })

        fields = field_info.index.dropna()
        for each_f in fields:

            type_name = DataCateNames.get(
                str(field_info.loc[each_f, FieldInfoNames.CATE]))
            assert type_name is not None, RuntimeError(
                f'invalid value of type name {type_name}')

            if type_name == DataCateNames.SET:
                type_ = 'set'
                range_ = set_defs[field_info.loc[each_f,
                                                 FieldInfoNames.CATE_CODE]]
                assert len(list(range_.keys())) == len(set(range_.keys())
                    ), f'got repeated code in {each_f}'
            elif type_name == DataCateNames.VALUE:
                ret = value_defs[field_info.loc[each_f,
                                                FieldInfoNames.CATE_CODE]]
                type_ = ret['dtype'].lower()
                range_ = ret['range']

            if field_info.loc[each_f, FieldInfoNames.NULLABLE] == 'N':
                nullable = False
            elif field_info.loc[each_f, FieldInfoNames.NULLABLE] == 'Y':
                nullable = True
            else:
                raise RuntimeError('undefined nullable value')

            methods = (AlgorithmCodes.REG,) if type_ != 'set' else (AlgorithmCodes.DTC,)

            col_info = {
                "table": each_t.target_file,
                "code": each_f,
                "label": field_info.loc[each_f, FieldInfoNames.LABEL],
                "column": field_info.loc[each_f, FieldInfoNames.NAME],
                "nullable": nullable,
                "dtype": type_,
                "range": range_,
                "methods": methods
            }
            table["table_fields"][each_f] = col_info

        configs[each_t.target_file] = table

    json.dump(configs, open(SCHEMA_CONFIG_LOC, 'w',
              encoding='utf-8'), ensure_ascii=False)
except Exception as esp:
    logging.error(traceback.format_exc())
