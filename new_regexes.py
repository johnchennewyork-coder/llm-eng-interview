import re

text = "Item 1: 1000, Item 2: 3000"

# NEW PATTERN: Two sets of parentheses
# 1. (\d)   -> Capture the Item ID
# 2. (\d+)  -> Capture the Price
pattern = r"(\d): (\d+)"

matches = list(re.finditer(pattern, text))

# --- Looking at the First Match (Item 1) ---
first_match = matches[0]

print(first_match.group(0)) # "1: 1000" (The whole matched string)
print(first_match.group(1)) # "1"       (The content of the 1st parenthesis)
print(first_match.group(2)) # "1000"    (The content of the 2nd parenthesis)

# --- Looking at the Second Match (Item 2) ---
second_match = matches[1]

print(second_match.group(1)) # "2"
print(second_match.group(2)) # "3000"