# -*- coding: utf-8 -*-
"""
Created on Thursday 10 20 22:33:09 2022

@author: Jeff
"""
import glob
import os
import re
from tabnanny import verbose

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
                new_line = each.split('#')[0]
                if new_line.strip(' '):
                    new_line = new_line.rstrip(' ')
                    if new_line[-1] != '\n':
                        new_line += '\n'
                    ret.append(new_line)
            else:
                ret.append(each)
    
    with open(py, 'w', encoding='utf-8') as f:
        f.writelines(ret)
        """ ''' """
