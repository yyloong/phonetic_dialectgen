import pandas as pd
import tqdm
import os
from Text_to_IPA import text_to_IPA
def process_csv(input_csv: str, input_col: int,languaga: str,save_path: str) -> str:
    """Process CSV file to add IPA."""
    print(f"读取 CSV 文件：{input_csv}")
    df = pd.read_csv(input_csv)
    if input_col >= len(df.columns):
        print(f"输入列索引 {input_col} 超出 CSV 列范围")
        raise ValueError(f"Input column index {input_col} out of range")

    ipa_results = []
    failed_words_list = []
    failed_rows = 0

    # Process each row
    for sentence in tqdm(df.iloc[:, input_col], desc="处理行"):
        if not isinstance(sentence, str):
            print(f"无效输入，非字符串: {sentence}")
            ipa_results.append("")
            failed_words_list.append([])
            failed_rows += 1
            continue

        ipa_str, failed_words, success = text_to_IPA(sentence,language=languaga)
        ipa_results.append(ipa_str)
        failed_words_list.append(",".join(failed_words))
        if not success:
            failed_rows += 1

    # Add IPA and FailedWord columns
    df["IPA"] = ipa_results
    df["FailedWord"] = failed_words_list

    # Save to new CSV
    df.to_csv(save_path, index=False, encoding="utf-8")

    failure_rate = failed_rows / len(df) if len(df) > 0 else 0
    print(
        f"{save_path}: 处理完成，失败行数: {failed_rows}/{len(df)}，失败率: {failure_rate:.2%}"
    )
    return save_path


