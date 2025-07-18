# 删除注释行
output = 'shupin.yaml'

with open('shupin.dict.yaml', 'r', encoding='utf-8') as file, open(output, 'w', encoding='utf-8') as out:
    for line in file:
        if line.startswith('#') or line.strip() == '':
            continue
        line = line.split('\t')
        if len(line) < 2:
            continue
        if len(line[0]) > 1:
            continue
        value = line[1].split('◎')[0].strip()
        if value[0] >= 'A' and value[0] <= 'Z':
            continue
        out.write(f'{line[0]}: {value}\n')
        