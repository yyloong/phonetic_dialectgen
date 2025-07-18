# 给 wav 文件批量重命名

import os
import pandas as pd

df = pd.read_csv("aitts3_shu.csv", encoding='utf-8')

for i in range(1, 3001):
    old_name = f"sichuan/{i}.wav"
    new_name = f"sichuan/{df.loc[i-1, 'audio']}.wav"
    
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        print(f"重命名: {old_name} -> {new_name}")
    else:
        print(f"文件不存在: {old_name}")


