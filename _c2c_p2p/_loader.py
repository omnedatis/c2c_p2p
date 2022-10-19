# -*- coding: utf-8 -*-
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd

from .common import (OUTPUT_LOC, SCHEMA_CONFIG_LOC, Intervals, SchemaTableRefs,
                    tableType, ExtendedColumn, SPLITER)

PK = 'customerid'
PK2 = 'prod_code'


class _LocalDataProvider:
    """取得訓練及驗證資料之物件"""
    def __init__(self) -> None:
        self._configs: dict = json.load(open(SCHEMA_CONFIG_LOC, 'r', encoding='utf-8'))

    def get_data(self, check: Optional[bool] = False, training: Optional[bool] = True):
        if not os.path.isdir(f'{OUTPUT_LOC}/{is_training}'):
            os.mkdir(f'{OUTPUT_LOC}/{is_training}')

        ret = {}
        is_training = 'train' if training else 'validate'
        for key, each in self._configs.items():

            logging.debug(f'Getting data for {key}')
            if os.path.isfile(f'{OUTPUT_LOC}/{is_training}/{each["table_name"]}.pkl'):
                data = pickle.load(open(f'{OUTPUT_LOC}/{is_training}/{each["table_name"]}.pkl', 'rb'))
            else:
                data = self._load(each['table_name'], training=training)
                if check:
                    data = self._check(data, each['table_name'],
                                    each['table_fields'])
                pickle.dump(data, open(f'{OUTPUT_LOC}/{is_training}/{each["table_name"]}.pkl', 'wb'))

            ret[each['table_name']] = data
            logging.info(f'Loading table {key} complete')
        return ret

    def _load(self, table_name: str, training: bool = True) -> pd.DataFrame:
        if training:
            fname = f'./{table_name}_训练.csv'
        else:
            fname = f'./{table_name}_验证.csv'
        data = pd.read_csv(fname, header=0, encoding='utf-8-sig')
        return data

    def _check(self, data: pd.DataFrame, table_name: str,
               table_info: tableType) -> pd.DataFrame:
        int_pat = re.compile(r'[+-]?\d+.?') # integer pattern
        float_pat = re.compile(r'[+-]?\d+(\.\d*)?') # float pattern
        float_pat2 = re.compile(r'[-+]?([0-9]*[.])?[0-9]+[eE][-+]?\d+')
        data_pat = (r'(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})' # !!! date pattern
                    r'(0[1-9]|1[012])'
                    r'(0[1-9]|[1-2][0-9]|3[01])')
        date_pat = re.compile(data_pat)

        # read column by definition (if exists)
        errors = []
        data_col = []
        error_col = []
        for each in table_info:
            if each in data.columns.tolist():
                logging.debug(f'Checking column {each}')
                data_col.append(each)
                series = data[each].values
                finfo = table_info[each]
                dtype = finfo['dtype']
                range_ = finfo['range']
                nullable = finfo['nullable']

                # nan mask
                if nullable:
                    nan = ~(series == series)
                else:
                    nan = (series == series)

                # !!!
                # check type with no bounds
                if dtype == 'set':
                    is_int = np.vectorize(lambda x: bool(int_pat.fullmatch(x)))
                    result = np.isin(series, range_)
                elif dtype == 'string':
                    result = np.full(series.shape, True)
                elif dtype == 'date':
                    is_date = np.vectorize(
                        lambda x: bool(date_pat.fullmatch(x)))
                    result = is_date(series.astype('str'))

                # check type with bounds
                else:
                    if dtype == 'integer':
                        is_int = np.vectorize(
                            lambda x: bool(int_pat.fullmatch(x)))
                        t_result = is_int(series.astype('str'))
                        cast = 'int'
                    elif dtype == 'float':
                        is_float = np.vectorize(lambda x: bool(
                            float_pat.fullmatch(x) or float_pat2.fullmatch(x)))
                        t_result = is_float(series.astype('str'))
                        cast = 'float'
                    else:
                        raise RuntimeError(f'data type {dtype} not understood')

                    # check range if types are all correct
                    if np.all(t_result):
                        for key, value in range_.items():
                            if value is not None:
                                if key == Intervals.LC:
                                    result = (value <= series.astype(cast))
                                elif key == Intervals.RC:
                                    result = (value >= series.astype(cast))
                                elif key == Intervals.LO:
                                    result = (value < series.astype(cast))
                                elif key == Intervals.RO:
                                    result = (value > series.astype(cast))
                    # else report type errors
                    else:
                        result = t_result
                # masking
                if nullable:
                    err = ~(result | nan)
                else:
                    err = ~(result & nan)
                errors.append(err)
                if np.any(err):
                    error_col.append(each)
            else:
                logging.debug(f'Column {each} not in data')
        # !!!
        # only summary now
        if np.any(np.array(errors)):
            errors = pd.DataFrame(np.array(errors).T, columns=data_col)
            errors.sum(axis=0).to_csv(f'{OUTPUT_LOC}/{table_name}_errors.csv')
            logging.error(
                f'Invalid data encountered in table {table_name}, on {error_col}'
            )
            raise RuntimeError('invalid data encountered')

        # !!!
        # change column names, and change types by definitions (if exists)
        new_dtype = {}
        new_col = {}
        for each in table_info.values():
            if each['code'] in data.columns:
                if each['dtype'] == 'set':
                    new_dtype[each['code']] = 'float'
                elif each['dtype'] == 'integer':
                    new_dtype[each['code']] = 'float'
                elif each['dtype'] == 'float':
                    new_dtype[each['code']] = 'float'
                elif each['dtype'] == 'string':
                    new_dtype[each['code']] = 'string'
                elif each['dtype'] == 'date':
                    new_dtype[each['code']] = 'datetime64[D]'
                else:
                    raise RuntimeError(
                        f'data type {each["code"]} not understood')
            if (each['code'] != PK) and (each['code'] != PK2):

                # !!!
                exc = ExtendedColumn(each['code'], each['t_name'],
                                     each['label'], each['name'],
                                     each['method'])
                new_col[each['code']] = SPLITER.join(exc)

        return data[data_col].astype(new_dtype).rename(columns=new_col)


class DataSet:
    def __init__(self):
        self._loader = _LocalDataProvider()
        self._training = self._loader.get_data(check=True, training=True)
        self._validate = self._loader.get_data(check=True, training=False)
        self._configs = json.load(open(SCHEMA_CONFIG_LOC, 'r', encoding='utf-8'))

    def p2p(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]
        if training:
            ret: pd.DataFrame = self._training[SchemaTableRefs.PROD_INFO.target_file]
        else:
            ret: pd.DataFrame = self._validate[SchemaTableRefs.PROD_INFO.target_file]
        ret = ret.set_index(PK2)

        x_columns = []
        y_columns = []
        columns = {i.split(SPLITER)[0]: i for i in ret.columns.to_list()}
        configs = self._configs[SchemaTableRefs.PROD_INFO.target_file]

        for each in configs['table_fields'].values():
            is_valid = (each['code'] != PK)
            is_valid &= (each['code'] != PK2)
            is_valid &= (each['code'] in columns)
            is_valid &= (each['dtype'] in ['float', 'integer', 'set'])
            if is_valid:
                col = columns[each['code']]
                if (each['label'] in x):
                    x_columns.append(col)
                if (each['label'] in y):
                    y_columns.append(col)
        return ret.loc[:, x_columns], ret.loc[:, y_columns]

    def c2c(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        if training:
            transaction: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TRANS.target_file]
            holdings: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_HODING.target_file]
            client: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TPYE.target_file]
        else:
            transaction: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TRANS.target_file]
            holdings: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_HODING.target_file]
            client: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TPYE.target_file]
        ret = transaction.merge(holdings,
                                on=[PK, PK2],
                                how='outer',
                                suffixes=('', '_'))
        ret = ret.merge(client, on=PK, how='right', suffixes=('', '_'))
        ret = ret.set_index(PK)

        x_columns = []
        y_columns = []
        columns = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in ret.columns.to_list()
        }
        configs = list(self._configs[SchemaTableRefs.CLIENT_TRANS.target_file]
                       ['table_fields'].values())
        configs += list(
            self._configs[SchemaTableRefs.CLIENT_HODING.target_file]
            ['table_fields'].values())
        configs += list(self._configs[SchemaTableRefs.CLIENT_TPYE.target_file]
                        ['table_fields'].values())

        for each in configs:
            is_valid = (each['code'] != PK)
            is_valid &= (each['code'] != PK2)
            is_valid &= (each['code'] in columns)
            is_valid &= (each['dtype'] in ['float', 'integer', 'set'])
            if is_valid:
                col = columns[each['code']]
                if (each['label'] in x):
                    x_columns.append(col)
                if (each['label'] in y):
                    y_columns.append(col)
        return ret.loc[:, x_columns], ret.loc[:, y_columns]

    def c2c_trans(self, x: Union[str, list], y: Union[str, list],
                  training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        if training:
            transaction: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TRANS.target_file]
            client: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TPYE.target_file]
        else:
            transaction: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TRANS.target_file]
            client: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TPYE.target_file]
        ret = transaction.merge(client, on=PK, how='inner', suffixes=('', '_'))
        ret = ret.set_index(PK)

        x_columns = []
        y_columns = []

        columns = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in ret.columns.to_list()
        }
        configs = list(self._configs[SchemaTableRefs.CLIENT_TRANS.target_file]
                       ['table_fields'].values())
        configs += list(self._configs[SchemaTableRefs.CLIENT_TPYE.target_file]
                        ['table_fields'].values())

        for each in configs:
            is_valid = (each['code'] != PK)
            is_valid &= (each['code'] in columns)
            is_valid &= (each['dtype'] in ['float', 'integer', 'set'])
            if is_valid:
                col = columns[each['code']]
                if (each['label'] in x):
                    x_columns.append(col)
                if (each['label'] in y):
                    y_columns.append(col)
        return ret.loc[:, x_columns], ret.loc[:, y_columns]

    def c2c_holding(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        if training:
            holdings: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_HODING.target_file]
            client: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TPYE.target_file]
        else:
            holdings: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_HODING.target_file]
            client: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TPYE.target_file]
        ret = client.merge(holdings, on=PK, how='inner', suffixes=('', '_'))
        ret = ret.set_index(PK)

        x_columns = []
        y_columns = []
        columns = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in ret.columns.to_list()
        }
        configs = list(self._configs[SchemaTableRefs.CLIENT_HODING.target_file]
                       ['table_fields'].values())
        configs += list(self._configs[SchemaTableRefs.CLIENT_TPYE.target_file]
                        ['table_fields'].values())

        for each in configs:
            is_valid = (each['code'] != PK)
            is_valid &= (each['code'] in columns)
            is_valid &= (each['dtype'] in ['float', 'integer', 'set'])
            if is_valid:
                col = columns[each['code']]
                if (each['label'] in x):
                    x_columns.append(col)
                if (each['label'] in y):
                    y_columns.append(col)
        return ret.loc[:, x_columns], ret.loc[:, y_columns]

    def c2c_client(self, x: Union[str, list], y: Union[str, list],
                   training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]
        if training:
            ret: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TPYE.target_file]
        else:
            ret: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TPYE.target_file]
        ret = ret.set_index(PK)

        x_columns = []
        y_columns = []
        columns = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in ret.columns.to_list()
        }
        configs = list(self._configs[SchemaTableRefs.CLIENT_TPYE.target_file]
                       ['table_fields'].values())
        for each in configs:
            is_valid = (each['code'] != PK)
            is_valid &= (each['code'] in columns)
            is_valid &= (each['dtype'] in ['float', 'integer', 'set'])
            if is_valid:
                col = columns[each['code']]
                if (each['label'] in x):
                    x_columns.append(col)
                if (each['label'] in y):
                    y_columns.append(col)
        return ret.loc[:, x_columns], ret.loc[:, y_columns]

    def gen_p2p(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]
        if training:
            prod: pd.DataFrame = self._training[SchemaTableRefs.PROD_INFO.target_file]
        else:
            prod: pd.DataFrame = self._validate[SchemaTableRefs.PROD_INFO.target_file]

        prod_col = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in prod.columns.tolist() if i != PK2
        }
        prod_configs = list(self._configs[
            SchemaTableRefs.PROD_INFO.target_file]['table_fields'].values())
        x_col = []
        for each in prod_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in x)
            if is_valid:
                x_col.append(prod_col[each['code']])
        y_col = []
        for each in prod_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in y)
            if is_valid:
                y_col.append(prod_col[each['code']])
        prod = prod.set_index(PK2)
        # !!!
        step = 10 if training else 1
        xdata, ydata = prod[x_col].iloc[::step], prod[y_col].iloc[::step]
        for each in ydata:
            yield xdata, ydata[each]

    def gen_c2c_client(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        if training:
            client: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TPYE.target_file]
        else:
            client: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TPYE.target_file]

        client_col = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in client.columns.tolist() if i != PK
        }
        client_configs = list(self._configs[
            SchemaTableRefs.CLIENT_TPYE.target_file]['table_fields'].values())
        x_col = []
        for each in client_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in x)
            if is_valid:
                x_col.append(client_col[each['code']])
        y_col = []
        for each in client_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in y)
            if is_valid:
                y_col.append(client_col[each['code']])
        client = client.set_index(PK)
        # !!!
        step = 5 if training else 1
        xdata, ydata = client[x_col].iloc[::step,], client[y_col].iloc[::step,]
        for each in ydata:
            yield xdata, ydata[each]

    def gen_c2c_trans( self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        if training:
            transaction: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TRANS.target_file]
            client: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TPYE.target_file]
        else:
            transaction: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TRANS.target_file]
            client: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TPYE.target_file]

        client_col = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in client.columns.tolist() if i != PK and i !=PK2
        }
        client_configs = list(self._configs[
            SchemaTableRefs.CLIENT_TPYE.target_file]['table_fields'].values())
        x_col = [PK]
        for each in client_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in x)
            if is_valid:
                x_col.append(client_col[each['code']])

        trans_col = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in transaction.columns.tolist() if i != PK and i != PK2
        }
        trans_configs = list(self._configs[
            SchemaTableRefs.CLIENT_TRANS.target_file]['table_fields'].values())
        y_col = [PK]
        for each in trans_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in y)
            if is_valid:
                y_col.append(trans_col[each['code']])
        # !!!
        step = 25 if training else 1
        xdata, ydata = client[x_col], transaction[y_col].iloc[::step,]
        for each in ydata:
            if each != PK:
                data = xdata.merge(ydata[[PK, each]],
                                   on=PK,
                                   how='inner',
                                   suffixes=('', '_')).set_index(PK)
                x_col = [i for i in x_col if i != PK]
                yield data[x_col], data[each]

    def gen_c2c_holding(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        if training:
            holdings: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_HODING.target_file]
            client: pd.DataFrame = self._training[SchemaTableRefs.CLIENT_TPYE.target_file]
        else:
            holdings: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_HODING.target_file]
            client: pd.DataFrame = self._validate[SchemaTableRefs.CLIENT_TPYE.target_file]

        client_col = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in client.columns.tolist() if i != PK and i !=PK2
        }
        client_configs = list(self._configs[
            SchemaTableRefs.CLIENT_TPYE.target_file]['table_fields'].values())
        x_col = [PK]
        for each in client_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in x)
            if is_valid:
                x_col.append(client_col[each['code']])

        trans_col = {
            ExtendedColumn(*i.split(SPLITER)).code: i
            for i in holdings.columns.tolist() if i != PK and i != PK2
        }
        trans_configs = list(self._configs[
            SchemaTableRefs.CLIENT_TRANS.target_file]['table_fields'].values())
        y_col = [PK]
        for each in trans_configs:
            is_valid = (each['dtype'] in ['float', 'integer', 'set'])
            is_valid &= (each['label'] in y)
            if is_valid:
                y_col.append(trans_col[each['code']])
        # !!!
        step = 10 if training else 1
        xdata, ydata = client[x_col], holdings[y_col].iloc[::step,]
        for each in ydata:
            if each != PK:
                data = xdata.merge(ydata[[PK, each]],
                                   on=PK,
                                   how='inner',
                                   suffixes=('', '_')).set_index(PK)
                x_col = [i for i in x_col if i != PK]
                yield data[x_col], data[each]
