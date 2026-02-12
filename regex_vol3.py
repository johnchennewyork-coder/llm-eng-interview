

import re 

# e.g. a hostname parser 
'''
http://abc.bac.org/qqq/we1
'''

pattern = re.compile(r'http://([\w\.]+)/?')

print(pattern)

names = [

    'http://abc.bac.org/qqq/we1',
    'http://vscode.microsoft.ai/qqq/we1',
]

for name in names:
    match_obj = re.match(pattern, name)
    if match_obj:
        print('matched', match_obj.group(1) )


# TODO:
# re.findall, re.finditer etc.



