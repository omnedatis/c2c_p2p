# -*- coding: utf-8 -*-
"""
Created on Thursday 10 20 22:33:09 2022

@author: Jeff
"""
import glob
import os
import re

py_files = [i for i in glob.glob('**/*.py', recursive=True) if i.split('\\')[0] != 'env']
ptn = '''
((?<!\')\'[^\'"]*".*"[^\'"]*\'(?!\')|
(?<!\')\'[^\'"]*\'(?!\')|
(?<!")"[^\'"]*\'.*\'[^\'"]*"(?!")|
(?<!")"[^\'"]*"(?!")|)
'''
str_pat = re.compile(ptn, flags=(re.VERBOSE | re.DOTALL))
docstring_ptn = re.compile('(\'{3}|"{3})')
for py in py_files:
    if py == os.path.basename(os.path.realpath(__file__)):
        continue
    ret = []
    with open(py, 'r', encoding='utf-8') as f:
        doctring_stack = []
        is_docstring = False
        for each in f:
            if each == '# -*- coding: utf-8 -*-\n':
                ret.append(each)
                continue
            stripped_line, sub_count = str_pat.subn('', each)
            docstrings = docstring_ptn.findall(stripped_line)
            is_docstring = bool(docstrings)
            for doc in docstrings:
                if not doctring_stack:
                    doctring_stack.append(doc)
                elif doc in doctring_stack:
                    doctring_stack.pop() 
                else:
                    pass
            if is_docstring:
                pass
            elif '#' in stripped_line and (not doctring_stack):
                cut = stripped_line[stripped_line.find('#'):]
                new_line = each[:-len(cut)].rstrip(' ')
                if new_line:
                    ret.append(new_line+'\n')
            else:
                ret.append(each)
    
    with open(py, 'w', encoding='utf-8') as f:
        f.writelines(ret)
