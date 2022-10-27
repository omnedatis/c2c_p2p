# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple
import datetime
import logging
import os
import traceback
from typing import Any, Dict, List, Callable, Generator, Tuple

import numpy as np
import pandas as pd
from pandas import ExcelWriter
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

from _c2c_p2p import (OUTPUT_LOC, SPLITER, ExtendedColumn, DataSet,
                      ColumnManager, AlgorithmCodes, Task, dataFuncType)
try:
    dataset = DataSet()
    tasks = [
        Task(['C_1'], ['C_2'], 'C_1 predict C_2', 'c2c_client'),
        Task(['C_1', 'C_2'], ['C_3'], 'C_1 and C_2 predict C_3', 'c2c_trans'),
        Task(['C_1', 'C_2'], ['C_4'], 'C_1 and C_2 predict C_4', 'c2c_holding'),
        Task(['P_1'], ['P_2', 'P_3', 'P_4'],
             'P_1 predict P_2 and P_3 and P_4', 'p2p'),
        Task(['P_1'], ['P_3'], 'P_1 predict P_3', 'p2p'),
        Task(['P_1'], ['P_4'], 'P_1 predict P_4', 'p2p')
    ]

    featureName = str
    targetName = str
    targetNames = List[targetName]

    begin = datetime.datetime.now()
    results = BufferList()
    for i, task in enumerate(tasks):

        logging.info(f'Start on task {task.name}')
        task_begin = datetime.datetime.now()

        # get dataset
        func: dataFuncType = eval(f'dataset.{task.task}')
        x_raw, y_raw = func(task.x, task.y)
        x_test_raw, y_test_raw = func(task.x, task.y, training=False)
        for j, y_col in enumerate(y_raw):
            logging.info(f'Start on column {y_raw[y_col].name}'
                         f' {j+1}/{len(y_raw.columns.tolist())}')
            col_begin = datetime.datetime.now()

            # get target
            targets: targetNames = y_test_raw.index.tolist()
            target_features: Dict[targetName, List[featureName]] = {}

            # get data value
            _y_value: np.ndarray = y_raw[y_col].values
            _x_value: np.ndarray = x_raw.values

            # remove entries without target
            x_value: np.ndarray = x_value[(y_value == y_value), :]
            y_value: np.ndarray = y_value[(y_value == y_value)]

            # remove ineffective columns
            x_value: np.ndarray = _x_value[:,
                                           (_x_value == _x_value).sum(axis=0) != 0]

            # get data column name
            target_features[y_raw[y_col].name] = x_raw.columns[(
                _x_value == _x_value).sum(axis=0) != 0].tolist()

            # only estimate when x has non-zero entries, non-zero fields, and y has any entry
            if x_value.shape[0] != 0 and x_value.shape[1] != 0 and y_value.shape[0] != 0:

                # get analysis methods
                methods = ColumnManager.get(y_raw[y_col].name).methods

                for each_m in methods:

                    # by case estimation
                    if each_m == AlgorithmCodes.REG:
                        logging.info('Perform regression')
                        pipe = Pipeline(steps=[
                            ('impute', SimpleImputer(missing_values=np.nan)),
                            ('tree', LinearRegression())
                        ])

                    elif each_m == AlgorithmCodes.DTC:

                        pipe = Pipeline(steps=[
                            ('impute', SimpleImputer(missing_values=np.nan)),
                            ('tree',  DecisionTreeClassifier(
                                criterion='entropy', max_depth=15))
                        ])

                        logging.info('Training decision tree')
                        pipe.fit(x_value, y_value)
                        logging.info('Training decision tree complete')

                        # corresponding feature columns and feature data (test)
                        f_cols = target_features[y_test_raw[y_col].name]
                        x_test = x_test_raw[f_cols]

                        # predicted weights (for all entries)
                        class_wieghts = pd.DataFrame(
                            pipe.predict_proba(x_test), index=x_test.index)

                        # target column information
                        y_info = ColumnManager.get(y_test_raw[y_col].name)

                        # output tree txt
                        tree = export_text(
                            pipe['tree'], feature_names=f_cols, show_weights=True)
                        with open(f'{OUTPUT_LOC}/trees/{y_info.code}_{task.name}.txt', 'w', encoding='utf-8') as f:
                            f.writelines(tree)

                        # output result for all entries
                        for k, each_t in enumerate(targets):
                            logging.debug((f'Predict task {task.name}:' 
                                           f'{i+1}/{len(tasks)},'
                                           f' target {j+1},' 
                                           f' entry {k+1}/{len(targets)}'))
                            w = class_wieghts.loc[each_t, ].values
                            if len(w.shape) > 1:
                                w = w.mean(axis=0)
                            orders = np.argsort(w)[::-1]
                            decoder = SetCodeManager.get(y_info.name).decode
                            line = [each_t, task.name, y_info.code, y_info.table, 
                                y_info.column]
                            line += [f'{decoder(pipe["tree"].classes_[o])}'
                                f'({round(w[o], 2)}%)' for o in orders]
                            results.append(line)

                col_time = datetime.datetime.now() - col_begin
                logging.info(f'Column {y_raw[y_col].name} complete, took {col_time}'
                             f' {j+1}/{len(y_raw.columns.tolist())}')
            else:
                logging.info(
                    f'Column {y_raw[y_col].name} skipped for invalid data')

        task_time = datetime.datetime.now() - task_begin
        logging.info(f'Task {task.name} completed, took {task_time}')

    total_time = datetime.datetime.now() - begin
    logging.info(f'All task finished, total time taken {total_time}')
    logging.info('Start writing file')
    for key_c, value_c in results.items():
        with ExcelWriter(f'{OUTPUT_LOC}/reports/{key_c}.xlsx') as writer:
            for key_t, value_t in value_c.items():
                data = pd.DataFrame.from_dict(value_t, orient='index')
                data.to_excel(writer, key_t)

except Exception:
    logging.error(traceback.format_exc())
