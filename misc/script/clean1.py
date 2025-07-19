import yaml
from opencc import OpenCC

with open('shupin.yaml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# 繁体字转换为简体字
cc = OpenCC('t2s')

with open('shupin_simp.yaml', 'w', encoding='utf-8') as out:
    for key, value in data.items():
        key = cc.convert(key)
        out.write(f'{key}: {value}\n')
        