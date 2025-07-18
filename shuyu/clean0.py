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

# import yaml

# with open('dict_cleaned.yaml', 'r', encoding='utf-8') as file:
#     data = yaml.safe_load(file)

# # 繁体字转换为简体字

# print(data.keys())

# from opencc import OpenCC

# cc = OpenCC('t2s')

# with open('dict_simp.yaml', 'w', encoding='utf-8') as out:
#     for i in range(len(data)):
        

# with open('7000.txt', 'r', encoding='utf-8') as file:
#     with open('7000_simp.txt', 'w', encoding='utf-8') as out:
#         for line in file:
#             line = line.strip()
#             for char in line:
#                 out.write(f'{char}\n')
        