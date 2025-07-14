import pandas as pd
import argparse
import logging
import os
import jieba
from pypinyin import lazy_pinyin, Style
from tqdm import tqdm
from typing import Tuple, List
import re

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/csv_to_pinyin.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_pinyin_with_segmentation(sentence: str) -> Tuple[str, List[str], bool]:
    """Segment Chinese sentence and convert to Pinyin, retaining original punctuation."""
    if not sentence.strip():
        logger.error("输入句子为空")
        return "", [], False

    # Filter out non-Chinese characters
    hanzi_pattern = re.compile(r"[\u4e00-\u9fff]")

    # Segment sentence using jieba
    words = list(jieba.cut(sentence, cut_all=False))
    logger.info(f"分词结果: {words}")

    pinyin_parts = []
    failed_words = []
    i = 0
    sentence_len = len(sentence)
    success = True

    while i < sentence_len:
        char = sentence[i]

        # Handle non-Chinese characters (e.g., punctuation)
        if not hanzi_pattern.match(char):
            pinyin_parts.append(char)  # Retain punctuation
            i += 1
            continue

        # Find the next word that starts at position i
        matched = False
        for word in words:
            if sentence[i : i + len(word)] == word:
                try:
                    # Get pinyin for the word
                    pinyin_list = lazy_pinyin(
                        word, style=Style.TONE3, neutral_tone_with_five=True
                    )

                    # Verify pinyin count matches character count
                    if len(pinyin_list) == len(word):
                        pinyin_parts.extend(
                            pinyin_list
                        )  # Add each character's pinyin
                        logger.info(
                            f"成功获取拼音: {word} -> {' '.join(pinyin_list)}"
                        )
                    else:
                        logger.warning(
                            f"拼音数量 ({len(pinyin_list)}) 与汉字数 ({len(word)}) 不匹配: {word}"
                        )
                        failed_words.append(word)
                        success = False
                        # Fallback to single character processing
                        char_pinyin_parts = []
                        for char in word:
                            char_pinyin = lazy_pinyin(
                                char,
                                style=Style.TONE3,
                                neutral_tone_with_five=True,
                            )
                            if char_pinyin and char_pinyin[0]:
                                char_pinyin_parts.append(char_pinyin[0])
                            else:
                                logger.warning(f"无法为单字 '{char}' 获取拼音")
                                failed_words.append(char)
                                success = False
                        pinyin_parts.extend(char_pinyin_parts)

                    i += len(word)
                    matched = True
                    break
                except Exception as e:
                    logger.error(f"处理词 '{word}' 失败: {e}")
                    failed_words.append(word)
                    success = False
                    i += len(word)
                    matched = True
                    break

        if not matched:
            logger.warning(f"无法匹配词 at position {i}: '{char}'，忽略")
            failed_words.append(char)
            i += 1

    # Join pinyin parts with spaces, retaining punctuation
    result = " ".join(pinyin_parts).strip()
    success = len(failed_words) == 0 and result.strip() != ""
    logger.info(
        f"最终拼音: {sentence} -> {result}, 失败词: {failed_words}, 成功: {success}"
    )
    return result, failed_words, success


def process_csv(input_csv: str, input_col: int) -> str:
    """Process CSV file to add Pinyin column."""
    logger.info(f"读取 CSV 文件：{input_csv}")

    try:
        df = pd.read_csv(input_csv)
        if input_col >= len(df.columns):
            logger.error(f"输入列索引 {input_col} 超出 CSV 列范围")
            raise ValueError(f"Input column index {input_col} out of range")

        pinyin_results = []
        failed_words_list = []
        failed_rows = 0

        # Process each row
        for sentence in tqdm(df.iloc[:, input_col], desc="处理行"):
            if not isinstance(sentence, str):
                logger.warning(f"无效输入，非字符串: {sentence}")
                pinyin_results.append("")
                failed_words_list.append([])
                failed_rows += 1
                continue

            pinyin_str, failed_words, success = get_pinyin_with_segmentation(
                sentence
            )
            pinyin_results.append(pinyin_str)
            failed_words_list.append(",".join(failed_words))
            if not success:
                failed_rows += 1

        # Add Pinyin and FailedWord columns
        df["Pinyin"] = pinyin_results
        df["FailedWord"] = failed_words_list

        # Save to new CSV
        output_csv = os.path.splitext(input_csv)[0] + "_with_pinyin.csv"
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
        description="Convert Chinese text in CSV to Pinyin."
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
