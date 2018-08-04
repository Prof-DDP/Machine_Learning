# Fixing starbucks-menu-nutrition-food csv

with open('starbucks-menu-nutrition-food.txt', 'r') as f:
    lines = f.readlines()
    f.close()

lines[0] = lines[0].split()
lines[0] = ''.join(lines[0])

split_lines = [line.split('\x00') for line in lines]
split_lines_fixed = [''.join(split_line) for split_line in split_lines]

with open('starbucks-menu-nutrition-food-updated.txt', 'w') as f:
    for line in split_lines_fixed:
        f.write(line)
    f.close()

