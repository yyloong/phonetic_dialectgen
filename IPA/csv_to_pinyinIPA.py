import pandas as pd
import argparse
import logging
import os
from pypinyin import lazy_pinyin, Style
from tqdm import tqdm
from typing import Tuple, List
import re
from pinyin import pinyin_to_ipa

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/csv_to_ipa_simplified.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_ipa_per_character(sentence: str) -> Tuple[str, List[str], bool]:
    """Convert each Chinese character to IPA, retaining punctuation."""
    if not sentence.strip():
        logger.error("输入句子为空")
        return "", [], False

    # Filter out non-Chinese characters
    hanzi_pattern = re.compile(r"[\u4e00-\u9fff]")

    ipa_parts = []
    failed_words = []
    success = True

    for char in sentence:
        if not hanzi_pattern.match(char):
            ipa_parts.append(char)  # Retain punctuation
            logger.info(f"保留标点: {char}")
            continue

        try:
            # Get pinyin for the character
            pinyin_list = lazy_pinyin(
                char, style=Style.TONE3, neutral_tone_with_five=True
            )
            if pinyin_list and pinyin_list[0]:
                pinyin = pinyin_list[0]
                try:
                    ipa = pinyin_to_ipa(pinyin)
                    ipa_parts.append(ipa)
                    logger.info(f"成功获取 IPA: {char} -> {pinyin} -> {ipa}")
                except ValueError as e:
                    logger.warning(f"无法转换拼音 '{pinyin}' 到 IPA: {e}")
                    failed_words.append(char)
                    success = False
                    ipa_parts.append(pinyin)  # Fallback to pinyin
            else:
                logger.warning(f"无法为字符 '{char}' 获取拼音")
                failed_words.append(char)
                success = False
                ipa_parts.append(char)  # Fallback to character
        except Exception as e:
            logger.error(f"处理字符 '{char}' 失败: {e}")
            failed_words.append(char)
            success = False
            ipa_parts.append(char)  # Fallback to character

    # Join IPA parts with spaces, retaining punctuation
    result = " ".join(ipa_parts).strip()
    success = len(failed_words) == 0 and result.strip() != ""
    logger.info(
        f"最终 IPA: {sentence} -> {result}, 失败字符: {failed_words}, 成功: {success}"
    )
    return result, failed_words, success


def process_csv(input_csv: str, input_col: int) -> str:
    """Process CSV file to add IPA column."""
    logger.info(f"读取 CSV 文件：{input_csv}")

    try:
        df = pd.read_csv(input_csv)
        if input_col >= len(df.columns):
            logger.error(f"输入列索引 {input_col} 超出 CSV 列范围")
            raise ValueError(f"Input column index {input_col} out of range")

        ipa_results = []
        failed_words_list = []
        failed_rows = 0

        # Process each row
        for sentence in tqdm(df.iloc[:, input_col], desc="处理行"):
            if not isinstance(sentence, str):
                logger.warning(f"无效输入，非字符串: {sentence}")
                ipa_results.append("")
                failed_words_list.append([])
                failed_rows += 1
                continue

            ipa_str, failed_words, success = get_ipa_per_character(sentence)
            ipa_results.append(ipa_str)
            failed_words_list.append(",".join(failed_words))
            if not success:
                failed_rows += 1

        # Add IPA and FailedWord columns
        df["IPA"] = ipa_results
        df["FailedWord"] = failed_words_list

        # Save to new CSV
        output_csv = os.path.splitext(input_csv)[0] + "_with_pinyinIPA.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info(f"已保存结果到 {output_csv}")

        failure_rate = failed_rows / len(df) if len(df) > 0 else 0
        logger.info(
            f"处理完成，失败行数: {failed_rows}/{len(df)}，失败率: {failure_rate:.2%}"
        )
        print(
            f"{output_csv}: 处理完成，失败行数: {failed_rows}/{len(df)}，失败率: {failure_rate:.2%}"
        )
        return output_csv

    except FileNotFoundError:
        logger.error(f"CSV 文件未找到：{input_csv}")
        raise FileNotFoundError(f"CSV file not found: {input_csv}")
    except Exception as e:
        logger.error(f"处理 CSV 文件时出错：{e}")
        raise RuntimeError(f"Failed to process CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chinese text in CSV to IPA, treating each character as a word."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--input-col",
        type=int,
        default=1,
        help="Column index for input text (default: 1)",
    )
    args = parser.parse_args()

    process_csv(args.csv, args.input_col)


if __name__ == "__main__":
    main()
