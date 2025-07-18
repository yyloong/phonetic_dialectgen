import yaml

with open('shupin_simp.yaml', 'r', encoding='utf-8') as file:
    mapping = yaml.safe_load(file)

def convert_text(text):
    converted_text = []
    for char in text:
        if char in mapping:
            converted_text.append(mapping[char])
        else:
            converted_text.append(char)
    return ' '.join(converted_text)

text = '我是一个测试文本'
converted_text = convert_text(text)
print(converted_text)
