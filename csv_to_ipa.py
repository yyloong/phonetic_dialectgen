import pandas as pd
import argparse
from tqdm import tqdm
from goruut_ipa import run_goruut_and_get_ipa
import logging
import os

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


def process_csv(csv_file: str, language: str, input_col: int) -> None:
    """Process a CSV file to add IPA phonemes for Chinese text in the specified column."""
    try:
        # Read CSV file
        logger.info(f"读取 CSV 文件：{csv_file}")
        df = pd.read_csv(csv_file)
        if input_col >= len(df.columns):
            logger.error(
                f"输入列索引 {input_col} 超出 CSV 列数 {len(df.columns)}"
            )
            raise ValueError(
                f"Input column index {input_col} exceeds number of columns {len(df.columns)}"
            )

        input_col_name = df.columns[input_col]
        logger.info(f"处理列：{input_col_name}")

        # Initialize new columns
        df["IPA"] = ""
        df["FailedWord"] = ""

        # Process each row starting from the first row
        failed_rows = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理行"):
            sentence = str(row[input_col_name]).strip()
            if not sentence:
                logger.warning(f"行 {idx} 为空，跳过")
                continue

            try:
                ipa, failed_words, success = run_goruut_and_get_ipa(
                    language, sentence
                )
                df.at[idx, "IPA"] = ipa
                df.at[idx, "FailedWord"] = (
                    ",".join(failed_words) if failed_words else ""
                )
                if not success:
                    failed_rows += 1
                    logger.warning(f"行 {idx} 处理失败，失败词: {failed_words}")
            except Exception as e:
                logger.error(f"行 {idx} 处理出错: {e}")
                df.at[idx, "IPA"] = ""
                df.at[idx, "FailedWord"] = sentence
                failed_rows += 1

        # Save results
        output_file = csv_file.replace(".csv", "_with_IPA.csv")
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"已保存结果到 {output_file}")

        # Log failure rate
        failure_rate = failed_rows / len(df) if len(df) > 0 else 0
        logger.info(
            f"处理完成，失败行数: {failed_rows}/{len(df)}，失败率: {failure_rate:.2%}"
        )
        print(
            f"{output_file}: 处理完成，失败行数: {failed_rows}/{len(df)}，失败率: {failure_rate:.2%}"
        )

    except FileNotFoundError:
        logger.error(f"CSV 文件未找到：{csv_file}")
        raise
    except Exception as e:
        logger.error(f"处理 CSV 时出错：{e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Chinese text in CSV to IPA phonemes."
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Input CSV file path"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese Mandarin",
        help="Language for phoneme conversion",
    )
    parser.add_argument(
        "--input-col",
        type=int,
        default=0,
        help="Column index containing Chinese text",
    )
    args = parser.parse_args()

    process_csv(args.csv, args.language, args.input_col)
