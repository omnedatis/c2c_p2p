# -*- coding: utf-8 -*-
import logging
import json
import traceback
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from common import (SCHEMA_CONFIG_LOC, typeNames, FieldInfoNames, SchemaTableRefs,
                    Intervals, configType)
try:
    # read all table schema
    configs: configType = {}
    for each_t in SchemaTableRefs:
        table = {
            "table_name": each_t.target_file,
            "table_fields": {},
        }
        # read main definition (field definitions)
        field_info: pd.DataFrame = pd.read_excel(
            each_t.file_name, each_t.main_def).set_index(FieldInfoNames.CODE.value)

        # read set definitions
        set_sheet = pd.read_excel(
            each_t.file_name, each_t.set_def, header=None).values
        set_elements = field_info.loc[field_info[FieldInfoNames.TYPE.value]
                                      == typeNames.SET.value][FieldInfoNames.SET_CODE.value].tolist()
        set_defs = {}
        for rid, row in enumerate(set_sheet):
            for cid, cell in enumerate(row.tolist()):
                if cell in set_elements:
                    set_defs[cell] = [
                        int(i) for i in set_sheet[rid+2].tolist() if i == i]

        # read value definitions
        value_sheet = pd.read_excel(each_t.file_name, each_t.value_def)
        column_names = {i: col for i, col in enumerate(value_sheet)}
        value_defs = {}
        for rid, row in enumerate(value_sheet.values):
            for cid, cell in enumerate(row):
                cname = column_names[cid]

                if cname == '名称':
                    cell_key = cell
                    value_defs[cell_key] = {
                        'type': '',
                        'intervals': {}
                    }
                    if cell_key is None:
                        break
                elif cname == '资料型态':
                    value_defs[cell_key]['type'] += cell.lower()

                elif cname == Intervals.LC.value:
                    value_defs[cell_key]['intervals'].update({
                        Intervals.LC.value: cell if (cell == cell) else None
                    })
                elif cname == Intervals.RC.value:
                    value_defs[cell_key]['intervals'].update({
                        Intervals.RC.value: cell if (cell == cell) else None
                    })
                elif cname == Intervals.RO.value:
                    value_defs[cell_key]['intervals'].update({
                        Intervals.RO.value: cell if (cell == cell) else None
                    })
                elif cname == Intervals.LO.value:
                    value_defs[cell_key]['intervals'].update({
                        Intervals.LO.value: cell if (cell == cell) else None
                    })

        # make unified schema for both value and set
        fields = field_info.index.dropna()
        for each_f in fields:

            # decide type and range for value and set
            type_name = typeNames.get(
                str(field_info.loc[each_f, FieldInfoNames.TYPE]))
            assert type_name is None, RuntimeError(
                f'invalid value of type name {type_name}')

            if type_name == typeNames.SET:
                type_ = 'set'
                range_ = set_defs[field_info.loc[each_f, FieldInfoNames.SET_CODE]]
            elif type_name == typeNames.VALUE:
                ret = value_defs[field_info.loc[each_f, FieldInfoNames.SET_CODE]]
                type_ = ret['type'].lower()
                range_ = ret['intervals']

            # whether is nullable
            if field_info.loc[each_f, FieldInfoNames.NULLABLE] == 'N':
                nullable = False
            elif field_info.loc[each_f, FieldInfoNames.NULLABLE] == 'Y':
                nullable = True
            else:
                raise RuntimeError('undefined nullable value')

            # !!!
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
