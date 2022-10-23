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
                      ColumnManager, AlgorithmCodes, Task, DataGeneratorType)
try:
    dataset = DataSet()
    tasks = [
        Task(['C_1'], ['C_2'], 'C_1 predict C_2', 'gen_c2c_client'),
        Task(['C_1', 'C_2'], ['C_3'], 'C_1 and C_2 predict C_3', 'gen_c2c_trans'),
        Task(['C_1', 'C_2'], ['C_4'], 'C_1 and C_2 predict C_4', 'gen_c2c_holding'),
        Task(['P_1'], ['P_2', 'P_3', 'P_4'],
            'P_1 predict P_2 and P_3 and P_4', 'gen_p2p'),
        Task(['P_1'], ['P_3'], 'P_1 predict P_3', 'gen_p2p'),
        Task(['P_1'], ['P_4'], 'P_1 predict P_4', 'gen_p2p')
    ]

    featureName = str
    targetName = str
    targetNames = List[targetName]


    begin = datetime.datetime.now()
    results = defaultdict(lambda: defaultdict(dict))
    for i, task in enumerate(tasks):

        logging.info(f'Start on task {task.name}')
        task_begin = datetime.datetime.now()

        func:DataGeneratorType = eval(f'dataset.{task.task}')
        for j, ((x_raw, y_raw), (x_test_raw, y_test_raw)) in enumerate(zip(func(task.x, task.y), func(task.x, task.y, training=False))):

            logging.info(f'Start on column {y_raw.name}')
            col_begin = datetime.datetime.now()

            targets: targetNames = y_test_raw.index.tolist()
            target_features: Dict[targetName, List[featureName]] = {}

            _y_value: np.ndarray = y_raw.values
            _x_value: np.ndarray = x_raw.values

            _x_value: np.ndarray = _x_value[(_y_value == _y_value), :]
            y_value: np.ndarray = _y_value[(_y_value == _y_value)]

            x_value: np.ndarray = _x_value[:,
                                        (_x_value == _x_value).sum(axis=0) != 0]

            target_features[y_raw.name] = x_raw.columns[(
                _x_value == _x_value).sum(axis=0) != 0].tolist()

            if x_value.shape[0] != 0 and x_value.shape[1] != 0 and y_value.shape[0] != 0:


                methods = ColumnManager.get(y_raw.name).methods
                
                for each_m in methods:

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

                        f_cols = target_features[y_test_raw.name]
                        x_test = x_test_raw[f_cols]

                        class_wieghts = pd.DataFrame(
                            pipe.predict_proba(x_test), index=x_test.index)

                        y_info = ColumnManager.get(y_test_raw.name)

                        tree = export_text(
                            pipe['tree'], feature_names=f_cols, show_weights=True)
                        with open(f'{OUTPUT_LOC}/trees/{y_info.c_name}_{task.name}.txt', 'w', encoding='utf-8') as f:
                            f.writelines(tree)

                        for k, each_t in enumerate(targets):
                            logging.debug((f'Predict task {task.name}: {i+1}/{len(tasks)},'
                                        f' target {j+1},' f' entry {k+1}/{len(targets)}'))
                            w = class_wieghts.loc[each_t, ].values
                            if len(w.shape) > 1:
                                w = w.mean(axis=0)
                            orders = np.argsort(w)[::-1]
                            results[each_t][task.name][y_info.code] = [y_info.t_name, y_info.c_name] \
                                + [f'{int(pipe["tree"].classes_[o])}({round(w[o], 2)}%)' for o in orders]

                col_time = datetime.datetime.now() - col_begin
                logging.info(f'Column {y_raw.name} complete, took {col_time}')
            else:
                logging.info(f'Column {y_raw.name} skipped for invalid data')

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
