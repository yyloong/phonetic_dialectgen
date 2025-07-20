import pandas as pd
from torchaudio import load
import tqdm

df = pd.read_csv('aitts3_shu.csv')

print(len(df))

# 3305, 7010

for i in tqdm.tqdm(range(len(df))):
    audio = df.loc[i, 'audio']
    # load(f'sichuan/{audio}.wav')
    try:
        load(f'sichuan/{audio}.wav')
    except:
        print(audio, end=',')