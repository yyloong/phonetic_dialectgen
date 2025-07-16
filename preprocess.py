import pandas as pd
import tqdm

file = pd.read_csv("edge-hk.csv")

# 根据观察 mel-spectrogram 
# 最好在每个句子的开头都加上"一个逗号和一个空格"
# 并且在末尾加上"一个空格和一个标点符号"
# 这样可以更好地训练 glow-tts 对时间的分割
for i in tqdm.tqdm(range(len(file))):
    file.loc[i, "IPA"] = '， ' + file.loc[i, "IPA"]
    punctuations = ',.!?;:，。！？；：'
    if file.loc[i, "IPA"][-1] not in punctuations:
        file.loc[i, "IPA"] += ' 。'

file.to_csv("edge-hk-new.csv", index=False)

# # 删除 "wav" 列
# file = file.drop(columns=["wav"])
# # 保存修改后的 DataFrame 到新的 CSV 文件
# file.to_csv("edge-hk1.csv", index=False)