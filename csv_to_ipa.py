import pandas as pd
import argparse
import logging
import os
from goruut_ipa import (
    run_goruut_and_get_ipa,
    get_ipa_with_segmentation,
    GoruutConfig,
    goruut_server_context,
)
from tqdm import tqdm

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/csv_to_ipa.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def process_csv(input_csv: str, language: str, input_col: int) -> str:
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
        config = GoruutConfig()

        # Start goruut server once before processing all rows
        with goruut_server_context(config) as server:
            # Process each row
            for sentence in tqdm(df.iloc[:, input_col], desc="处理行"):
                if not isinstance(sentence, str):
                    logger.warning(f"无效输入，非字符串: {sentence}")
                    ipa_results.append("")
                    failed_words_list.append([])
                    failed_rows += 1
                    continue

                # Call get_ipa_with_segmentation directly with server config
                ipa, failed_words, success = get_ipa_with_segmentation(
                    language, sentence, server.config
                )
                ipa_results.append(ipa)
                failed_words_list.append(",".join(failed_words))
                if not success:
                    failed_rows += 1

        # Add IPA and FailedWord columns
        df["IPA"] = ipa_results
        df["FailedWord"] = failed_words_list

        # Save to new CSV
        output_csv = os.path.splitext(input_csv)[0] + "_with_IPA.csv"
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
        description="Convert Chinese text in CSV to IPA."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--language",
        default="Chinese Mandarin",
        help="Language for IPA conversion (default: Chinese Mandarin)",
    )
    parser.add_argument(
        "--input-col",
        type=int,
        default=1,
        help="Column index for input text (default: 1)",
    )
    args = parser.parse_args()

    process_csv(args.csv, args.language, args.input_col)


if __name__ == "__main__":
    main()
