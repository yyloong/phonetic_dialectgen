import pandas as pd
from tokenizer import TTSTokenizer

csv_file = "aitts5.csv"
error_count = 0
valid_rows = []
file = pd.read_csv(csv_file)
tokenizer = TTSTokenizer()
for i in range(len(file)):
    try:
        token_ids = tokenizer(file["IPA"][i])
        if len(token_ids) > 0:
            valid_rows.append(file.iloc[i])
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        error_count += 1
        continue

print(f"Total errors encountered: {error_count}")
if valid_rows:
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv("cleaned.csv", index=False)
    print(f"Total valid rows: {len(valid_rows)}")
    print(f"Cleaned data saved to 'cleaned.csv'")
else:
    print("No valid rows found. No data saved.")
