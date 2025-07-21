import subprocess
import pandas as pd
import tqdm
import time

# pip install edge-tts

file = pd.read_csv("cv-hk.csv")

for i in tqdm.tqdm(range(len(file))):
    text = file.iloc[i]["text"]
    command = f"edge-tts --voice zh-HK-HiuMaanNeural --text '{text}' --write-media edge/{i+1}.wav"
    subprocess.run(command, shell=True)
    file.loc[i, "wav"] = f"{i+1}"
    time.sleep(0.1)

file.to_csv("edge-hk.csv", index=False)