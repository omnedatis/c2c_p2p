# -*- coding: utf-8 -*-
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd

from .common import (OUTPUT_LOC, SCHEMA_CONFIG_LOC, SPLITER, PK, PK2, ValueColumns,
                     SchemaTableRefs, TableNames, Dtypes, DataCateNames,
                     ExtendedColumn, ColumnManager, configType)


class _LocalDataProvider:

    def __init__(self) -> None:
        self._configs: configType = json.load(
            open(SCHEMA_CONFIG_LOC, 'r', encoding='utf-8'))

    def get_data(self, training: Optional[bool] = True):

        is_training = 'train' if training else 'validate'
        if not os.path.isdir(f'{OUTPUT_LOC}/{is_training}'):
            os.mkdir(f'{OUTPUT_LOC}/{is_training}')

        ret = {}
        for table_name in self._configs:
            logging.info(f'Reading table for {table_name}')
            if os.path.isfile(f'{OUTPUT_LOC}/{is_training}/{table_name}.pkl'):
                data = pickle.load(
                    open(f'{OUTPUT_LOC}/{is_training}/{table_name}.pkl', 'rb'))
            else:
                data = self._load(table_name, training=training)
                data = self._check(data, table_name)
                data = self._cast(data, table_name)
                pickle.dump(data, open(
                    f'{OUTPUT_LOC}/{is_training}/{table_name}.pkl', 'wb'))

            ret[table_name] = data
            logging.info(f'Reading table {table_name} complete')
        return ret

    def _load(self, table_name: str, training: bool = True) -> pd.DataFrame:
        if training:
            fname = f'./{table_name}_训练.csv'
        else:
            fname = f'./{table_name}_验证.csv'
        data = pd.read_csv(fname, header=0, encoding='utf-8-sig')
        return data

    def _check(self, data: pd.DataFrame, table_name: str) -> pd.DataFrame:
        int_pat = re.compile(r'[+-]?\d+.?')
        float_pat = re.compile(r'[+-]?\d+(\.\d*)?')
        float_pat2 = re.compile(r'[-+]?([0-9]*[.])?[0-9]+[eE][-+]?\d+')
        data_pat = (r'(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})'
                    r'(0[1-9]|1[012])'
                    r'(0[1-9]|[1-2][0-9]|3[01])')
        date_pat = re.compile(data_pat)

        table_info = self._configs[table_name]['table_fields']
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

                if nullable:
                    nan = ~(series == series)
                else:
                    nan = (series == series)

                if dtype == Dtypes.SET.value:
                    result = np.isin(series, range_)
                elif dtype == Dtypes.STRING.value:
                    result = np.full(series.shape, True)
                elif dtype == Dtypes.DATE.value:
                    is_date = np.vectorize(
                        lambda x: bool(date_pat.fullmatch(x)))
                    result = is_date(series.astype('str'))

                else:
                    if dtype == Dtypes.INT.value:
                        is_int = np.vectorize(
                            lambda x: bool(int_pat.fullmatch(x)))
                        t_result = is_int(series.astype('str'))
                        cast = 'float'
                    elif dtype == Dtypes.FLOAT.value:
                        is_float = np.vectorize(lambda x: bool(
                            float_pat.fullmatch(x) or float_pat2.fullmatch(x)))
                        t_result = is_float(series.astype('str'))
                        cast = 'float'
                    else:
                        raise RuntimeError(f'data type {dtype} not understood')

                    if np.all(t_result):
                        for key, value in range_.items():
                            if value is not None:
                                if key == ValueColumns.LC:
                                    result = (value <= series.astype(cast))
                                elif key == ValueColumns.RC:
                                    result = (value >= series.astype(cast))
                                elif key == ValueColumns.LO:
                                    result = (value < series.astype(cast))
                                elif key == ValueColumns.RO:
                                    result = (value > series.astype(cast))
                    else:
                        result = t_result
                if nullable:
                    err = ~(result | nan)
                else:
                    err = ~(result & nan)
                errors.append(err)
                if np.any(err):
                    error_col.append(each)
            else:
                logging.debug(f'Column {each} not in data')
        if np.any(np.array(errors)):
            errors = pd.DataFrame(np.array(errors).T, columns=data_col)
            errors.sum(axis=0).to_csv(f'{OUTPUT_LOC}/{table_name}_errors.csv')
            logging.error(
                f'Invalid data encountered in table {table_name}, on {error_col}'
            )
            raise RuntimeError('invalid data encountered')

        return data[data_col]

    def _cast(self, data: pd.DataFrame, table_name: str) -> pd.DataFrame:
        ColumnManager.load()
        table_info = self._configs[table_name]['table_fields']
        new_data = []
        for each in table_info:
            finfo = table_info[each]
            ex_col = ExtendedColumn(**finfo)
            if ex_col.code in data.columns:
                if ex_col.dtype == Dtypes.INT.value:
                    new_data.append(data[each].astype(
                        'float32').rename(ex_col.key))
                    ColumnManager.register(ex_col.key, ex_col)
                elif ex_col.dtype == Dtypes.FLOAT.value:
                    new_data.append(data[each].astype(
                        'float32').rename(ex_col.key))
                    ColumnManager.register(ex_col.key, ex_col)
                elif ex_col.dtype == Dtypes.SET.value:
                    cols: np.ndarray = np.array(ex_col.range)
                    values = data[each].values
                    one_hot = np.full((values.shape[0], cols.shape[0]), 0)
                    values, cols = np.ix_(values, cols)
                    one_hot[values == cols] = 1
                    names = [ExtendedColumn(**{
                        'code':ex_col.code+f'_{i}',
                        't_name':ex_col.t_name,
                        'c_name':ex_col.c_name+f'_{DataCateNames.SET}{i}',
                        'label':ex_col.label,
                        'nullable':ex_col.nullable,
                        "dtype": 'integer',
                        "range": [0, 1],
                        "methods": ex_col.methods
                    }) for i in ex_col.range]
                    for n in names:
                        ColumnManager.register(n.key, n)
                    names = [i.key for i in names]
                    new_data.append(pd.DataFrame(
                        one_hot, columns=names).astype('float32'))
                elif ex_col.code in [PK, PK2]:
                    new_data.append(data[each].astype('str'))
                    ColumnManager.register(ex_col.code, ex_col)
                else:
                    raise RuntimeError(
                        f'data type {finfo["code"]} not understood')
        ColumnManager.dump()
        return pd.concat(new_data, axis=1)


class _RandomDataProvider:
    def __init__(self, train_table_sizes: Optional[Dict[Union[str, TableNames], int]] = None,
                 valid_table_sizes: Optional[Dict[Union[str, TableNames], int]] = None,
                 random_id: Optional[bool] = None) -> None:
        self._configs: configType = json.load(
            open(SCHEMA_CONFIG_LOC, 'r', encoding='utf-8'))
        self._random_id = False
        if random_id is not None:
            self._random_id = random_id
        self._train_table_sizes = {
            TableNames.CLIENT_TPYE: 248000,
            TableNames.CLIENT_HODING: 619000,
            TableNames.CLIENT_TRANS: 3540000,
            TableNames.PROD_INFO: 550
        }
        if train_table_sizes is not None:
            self._train_table_sizes = train_table_sizes
        self._valid_table_sizes = {
            TableNames.CLIENT_TPYE: 248000,
            TableNames.CLIENT_HODING: 619000,
            TableNames.CLIENT_TRANS: 3540000,
            TableNames.PROD_INFO: 550
        }
        if valid_table_sizes is not None:
            self._valid_table_sizes = valid_table_sizes

    def get_data(self, training: Optional[bool] = True, write_local: Optional[bool] = False):

        is_training = 'train' if training else 'validate'
        if not os.path.isdir(f'{OUTPUT_LOC}/random_{is_training}') and write_local:
            os.mkdir(f'{OUTPUT_LOC}/random_{is_training}')

        ret = {}
        for table_name in self._configs:
            logging.debug(f'Generating data for {table_name}')
            if os.path.isfile(f'{OUTPUT_LOC}/random_{is_training}/{table_name}.csv'):
                data = pd.read_csv(f'{OUTPUT_LOC}/random_{is_training}/{table_name}.csv',
                                   header=0, encoding='utf-8-sig')
            else:
                sizes = self._train_table_sizes[table_name] if training else self._valid_table_sizes[table_name]
                data: pd.DataFrame = self.gen_random_sample(
                    table_name, sizes, self._random_id)
                data.to_csv(
                    f'{OUTPUT_LOC}/random_{is_training}/{table_name}.csv', encoding='utf-8-sig')

            ret[table_name] = data
            logging.info(f'Generating data {table_name} complete')
        return ret

    def gen_random_sample(self, table_name: str, sample_size: int,
                          random_id: bool) -> pd.DataFrame:
        if table_name not in self._configs:
            raise KeyError(f'{table_name} not found')

        ret = {}
        table = self._configs[table_name]['table_fields']
        for f_name, finfo in table.items():
            dtype = finfo['dtype']
            if dtype == Dtypes.FLOAT.value:
                left = 0
                right = 100
                for k, interval in finfo['range'].items():
                    if interval is not None:
                        if k == ValueColumns.LC.value:
                            left = interval
                        elif k == ValueColumns.LO.value:
                            left = interval + 1e-10
                        elif k == ValueColumns.RC.value:
                            right = interval
                        elif k == ValueColumns.RO.value:
                            right = interval - 1e-10
                ret[f_name] = np.random.rand(sample_size)*(right-left)-left
            elif dtype == Dtypes.INT.value:
                left = 0
                right = 100
                for k, interval in finfo['range'].items():
                    if interval is not None:
                        if k == ValueColumns.LC.value:
                            left = interval
                        elif k == ValueColumns.LO.value:
                            left = interval + 1
                        elif k == ValueColumns.RC.value:
                            right = interval
                        elif k == ValueColumns.RO.value:
                            right = interval - 1
                ret[f_name] = np.random.randint(left, right, sample_size)
            elif dtype == Dtypes.SET.value:
                set_ = finfo['range']
                ret[f_name] = np.random.choice(set_, sample_size, replace=True)
            elif finfo['code'] == PK:
                if random_id:
                    sample_id = np.array(
                        [f'CUST_{i+1}' for i in range(sample_size)*2])
                    np.random.shuffle(sample_id)
                    ret[f_name] = sample_id[::2]
                else:
                    ret[f_name] = np.array(
                        [f'CUST_{i+1}' for i in range(sample_size)])
            elif finfo['code'] == PK2:
                if random_id:
                    sample_id = np.array(
                        [f'PROD_{i+1}' for i in range(sample_size)*2])
                    np.random.shuffle(sample_id)
                    ret[f_name] = sample_id[::2]
                else:
                    ret[f_name] = np.array(
                        [f'PROD_{i+1}' for i in range(sample_size)])
        return pd.DataFrame(ret)


class DataSet:
    def __init__(self):
        self._loader = _LocalDataProvider()
        self._training = self._loader.get_data(training=True)
        self._validate = self._loader.get_data(training=False)
        self._current = self._training
        self._configs = json.load(
            open(SCHEMA_CONFIG_LOC, 'r', encoding='utf-8'))
        ColumnManager.load()

    def _set_db(self, training: bool):
        if training:
            self._current = self._training
        else:
            self._current = self._validate

    def p2p(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        prod: pd.DataFrame = self._current[SchemaTableRefs.PROD_INFO.target_file]
        prod = prod.set_index(PK2)

        columns = [i for i in prod.columns.tolist() if i != PK]
        x_columns = [i for i in columns if ExtendedColumn(
            *i.split(SPLITER)).label in x]
        y_columns = [i for i in columns if ExtendedColumn(
            *i.split(SPLITER)).label in y]
        return prod[x_columns], prod[y_columns]

    def c2c(self, x: Union[str, list], y: Union[str, list],
            training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        transactions: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TRANS.target_file]
        holdings: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_HODING.target_file]
        client: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TPYE.target_file]

        ret = transactions.merge(
            holdings, on=[PK, PK2], how='outer', suffixes=('', '_'))
        ret = ret.merge(client, on=PK, how='right', suffixes=('', '_'))
        ret = ret.set_index(PK)

        columns = [i for i in ret.columns.tolist() if i != PK2 and i !=
                   PK+'_' and i != PK2+'_']
        x_columns = [i for i in columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in columns if ColumnManager.get(i).label in y]

        return ret[x_columns], ret[y_columns]

    def c2c_client(self, x: Union[str, list], y: Union[str, list],
                   training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        client: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TPYE.target_file]
        client = client.set_index(PK)

        columns = [i for i in client.columns.tolist()]
        x_columns = [i for i in columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in columns if ColumnManager.get(i).label in y]

        return client[x_columns], client[y_columns]

    def c2c_trans(self, x: Union[str, list], y: Union[str, list],
                  training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        transactions: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TRANS.target_file]
        client: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TPYE.target_file]

        ret = transactions.merge(client, on=PK, how='inner', suffixes=('', '_'))
        ret = ret.set_index(PK)

        columns = [i for i in ret.columns.tolist() if i != PK2 and i != PK+'_']
        x_columns = [i for i in columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in columns if ColumnManager.get(i).label in y]

        return ret[x_columns], ret[y_columns]

    def c2c_holding(self, x: Union[str, list], y: Union[str, list],
                    training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        holdings: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_HODING.target_file]
        client: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TPYE.target_file]

        ret = client.merge(holdings, on=PK, how='inner', suffixes=('', '_'))
        ret = ret.set_index(PK)

        columns = [i for i in ret.columns.tolist() if i != PK2 and i != PK+'_']
        x_columns = [i for i in columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in columns if ColumnManager.get(i).label in y]

        return ret[x_columns], ret[y_columns]

    def gen_p2p(self, x: Union[str, list], y: Union[str, list],
                training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        prod: pd.DataFrame = self._current[SchemaTableRefs.PROD_INFO.target_file]
        prod = prod.set_index(PK2)

        columns = [i for i in prod.columns.tolist() if i != PK]
        x_columns = [i for i in columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in columns if ColumnManager.get(i).label in y]

        step = 10 if training else 1
        xdata, ydata = prod[x_columns].iloc[::step], prod[y_columns].iloc[::step]
        for each in ydata:
            yield xdata, ydata[each]

    def gen_c2c_client(self, x: Union[str, list], y: Union[str, list],
                       training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        client: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TPYE.target_file]
        client = client.set_index(PK)

        columns = [i for i in client.columns.tolist()]
        x_columns = [i for i in columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in columns if ColumnManager.get(i).label in y]

        step = 10 if training else 1
        xdata, ydata = client[x_columns].iloc[::step,
                                              ], client[y_columns].iloc[::step, ]
        for each in ydata:
            yield xdata, ydata[each]

    def gen_c2c_trans(self, x: Union[str, list], y: Union[str, list],
                      training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        transaction: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TRANS.target_file]
        client: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TPYE.target_file]

        x_columns = [i for i in client.columns.tolist()]
        x_columns = [i for i in x_columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in transaction.columns.tolist() if i != PK2]
        y_columns = [i for i in y_columns if ColumnManager.get(i).label in x]

        step = 25 if training else 1
        xdata, ydata = client[x_columns], transaction[y_columns].iloc[::step, ]
        for each in ydata:
            if each != PK:
                data = xdata.merge(
                    ydata[[each]], on=PK, how='inner', suffixes=('', '_')).set_index(PK)
                x_col = [i for i in data if i != each]
                print(x_col)
                yield data[x_col], data[each]

    def gen_c2c_holding(self, x: Union[str, list], y: Union[str, list],
                        training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        self._set_db(training)
        holdings: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_HODING.target_file]
        client: pd.DataFrame = self._current[SchemaTableRefs.CLIENT_TPYE.target_file]

        x_columns = [i for i in client.columns.tolist()]
        x_columns = [i for i in x_columns if ColumnManager.get(i).label in x]
        y_columns = [i for i in holdings.columns.tolist() if i != PK2]
        y_columns = [i for i in y_columns if ColumnManager.get(i).label in x]

        step = 10 if training else 1
        xdata, ydata = client[y_columns], holdings[y_columns].iloc[::step, ]
        for each in ydata:
            if each != PK:
                data = xdata.merge(
                    ydata[[PK, each]], on=PK, how='inner', suffixes=('', '_')).set_index(PK)
                x_col = [i for i in data if i != each]
                yield data[x_col], data[each]


class RandomDataSet(DataSet):

    def __init__(self, train_table_size: Optional[Dict[Union[str, TableNames], int]] = None,
                 valid_table_size: Optional[Dict[Union[str, TableNames], int]] = None):

        self._loader = _RandomDataProvider(train_table_size, valid_table_size)
        self._training = self._loader.get_data(training=True, write_local=True)
        self._validate = self._loader.get_data(training=False, write_local=True)
        self._current = self._training
        self._configs = json.load(
            open(SCHEMA_CONFIG_LOC, 'r', encoding='utf-8'))
