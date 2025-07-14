import pandas as pd

csv_file = "data_with_pinyinIPA.csv"
file = pd.read_csv(csv_file)

# 检测 file["text"] 是否包含英文字母
if file["text"].str.contains(r'[a-zA-Z]').any():
    print("文件包含英文字母")
else:
    print("文件不包含英文字母")

# print 包含英文字母的行
contains_english = file[file["text"].str.contains(r'[a-zA-Z]')]
print(contains_english)