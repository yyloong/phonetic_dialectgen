import pandas as pd
import yaml

with open('shupin.yaml', 'r', encoding='utf-8') as file:
    mapping = yaml.safe_load(file)

def convert_text(text):
    converted_text = []
    for char in text:
        if char in mapping:
            converted_text.append(mapping[char])
        else:
            converted_text.append(char)
    return ' '.join(converted_text)

df = pd.read_csv("aitts3.csv")

rows = []

for index, row in df.iterrows():
    converted_text = convert_text(row['text'])
    new_row = row[:-1].copy()
    new_row['IPA'] = converted_text
    rows.append(new_row)

new_df = pd.DataFrame(rows)
new_df.to_csv("aitts3_shu.csv", index=False, encoding='utf-8')