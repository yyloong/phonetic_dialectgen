import pandas as pd

df = pd.read_csv('aitts3_shu.csv')

# Add two blank spaces at the beginning of each 'IPA' entry
for i in range(len(df)):
    df.loc[i, 'IPA'] = '  ' + df.loc[i, 'IPA']

df.to_csv('blank_space.csv', index=False)