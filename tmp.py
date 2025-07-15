import pandas as pd
from tokenizer import TTSTokenizer

csv_file = "data_with_pinyinIPA.csv"
file = pd.read_csv(csv_file)
tokenizer = TTSTokenizer()
for i in range(len(file)):
    print(file["audio"][i])
    token_ids = tokenizer(file["IPA"][i])
