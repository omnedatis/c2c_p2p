# -*- coding: utf-8 -*-
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
single_eql_ptn = re.compile('(?<!=)=(?!=)')
for py in py_files:
    if py == os.path.basename(os.path.realpath(__file__)):
        continue
    ret = []
    with open(py, 'r', encoding='utf-8') as f:
        doctring_stack = []
        var_lines = None
        for each in f:
            if each == '# -*- coding: utf-8 -*-\n':
                ret.append(each)
                continue
            leave = False
            stripped_line, sub_count = str_pat.subn('', each)
            docstrings = docstring_ptn.findall(stripped_line)
            quotes = docstrings
            for doc in docstrings:
                if not doctring_stack:
                    doctring_stack.append(doc)
                    if bool(single_eql_ptn.search(each.split(docstrings[0])[0])):
                        var_lines = []
                elif doc in doctring_stack:
                    doctring_stack.pop()
                    if var_lines is not None:
                        ret.extend(var_lines+[each])
                        var_lines = None
                    leave = True
                else:
                    continue
            if doctring_stack:
                if var_lines is not None:
                    var_lines.append(each)
            else:
                if leave:
                    continue
                elif quotes:
                    if bool(single_eql_ptn.search(each.split(docstrings[0])[0])):
                        ret.append(each)
                    else:
                        ret.append(each.split(docstrings[0])[0])
                elif '#' in stripped_line:
                    cut = stripped_line[stripped_line.find('#'):]
                    new_line = each[:-len(cut)].rstrip(' ')
                    if new_line:
                        ret.append(new_line+'\n')
                else:
                    ret.append(each)
    
    with open(py, 'w', encoding='utf-8') as f:
        f.writelines(ret)
