# -*- coding: utf-8 -*-
import logging
import json
import traceback
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .common import (SCHEMA_CONFIG_LOC, DataCateNames, FieldInfoNames, SchemaTableRefs,
                    ValueColumns, configType)
try:
    configs: configType = {}
    for each_t in SchemaTableRefs:
        table = {
            "table_name": each_t.target_file,
            "table_fields": {},
        }
        field_info: pd.DataFrame = pd.read_excel(
            each_t.file_name, each_t.main_def).set_index(FieldInfoNames.CODE.value)

        set_sheet = pd.read_excel(
            each_t.file_name, each_t.set_def, header=None).values
        set_elements = field_info.loc[field_info[FieldInfoNames.CATE.value]
                                      == DataCateNames.SET.value][FieldInfoNames.CATE_CODE.value].tolist()
        set_defs = {}
        for rid, row in enumerate(set_sheet):
            for cid, cell in enumerate(row.tolist()):
                if cell in set_elements:
                    set_defs[cell] = [
                        int(i) for i in set_sheet[rid+2].tolist() if i == i]

        value_sheet = pd.read_excel(each_t.file_name, each_t.value_def)
        column_names = {i: col for i, col in enumerate(value_sheet)}
        value_defs = {}
        for rid, row in enumerate(value_sheet.values):
            for cid, cell in enumerate(row):
                cname = column_names[cid]

                if cname == ValueColumns.NAME.value:
                    cell_key = cell
                    value_defs[cell_key] = {
                        'type': '',
                        'intervals': {}
                    }
                    if cell_key is None:
                        break
                elif cname == ValueColumns.VALUEDTYPE.value:
                    value_defs[cell_key]['type'] += cell.lower()

                elif cname == ValueColumns.LC.value:
                    value_defs[cell_key]['intervals'].update({
                        ValueColumns.LC.value: cell if (cell == cell) else None
                    })
                elif cname == ValueColumns.RC.value:
                    value_defs[cell_key]['intervals'].update({
                        ValueColumns.RC.value: cell if (cell == cell) else None
                    })
                elif cname == ValueColumns.RO.value:
                    value_defs[cell_key]['intervals'].update({
                        ValueColumns.RO.value: cell if (cell == cell) else None
                    })
                elif cname == ValueColumns.LO.value:
                    value_defs[cell_key]['intervals'].update({
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
                range_ = set_defs[field_info.loc[each_f, FieldInfoNames.CATE_CODE]]
            elif type_name == DataCateNames.VALUE:
                ret = value_defs[field_info.loc[each_f, FieldInfoNames.CATE_CODE]]
                type_ = ret['type'].lower()
                range_ = ret['intervals']

            if field_info.loc[each_f, FieldInfoNames.NULLABLE] == 'N':
                nullable = False
            elif field_info.loc[each_f, FieldInfoNames.NULLABLE] == 'Y':
                nullable = True
            else:
                raise RuntimeError('undefined nullable value')

            method = '迴歸' if type_ != 'set' else '分類'

            table["table_fields"][each_f] = {
                "t_name": each_t.target_file,
                "code": each_f,
                "label": field_info.loc[each_f, FieldInfoNames.LABEL],
                "name": field_info.loc[each_f, FieldInfoNames.NAME],
                "nullable": nullable,
                "dtype": type_,
                "range": range_,
                "method": method
            }

        configs[each_t.target_file] = table

    json.dump(configs, open(SCHEMA_CONFIG_LOC, 'w',
              encoding='utf-8'), ensure_ascii=False)
except Exception as esp:
    logging.error(traceback.format_exc())
