

import re 


doc = 'hello my name is John'

receipts = 'Item 1: 1000, Item 2: 3000'
amt_extractor = r'\d: (\d+)'
match_obj = re.finditer(amt_extractor, receipts)

for match in match_obj:
    print(match)

# if match_obj:
#     print(match_obj.group(0))
#     print(match_obj.group(1))


# match_obj = re.search('John', doc)

# if match_obj:
#     print(match_obj.group(0))
#     # print(match_obj.group(1))

