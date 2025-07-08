import pandas as pd
import argparse
import os
import logging
from tqdm import tqdm
from goruut_ipa import get_ipa_phonemes, goruut_server_context, GoruutConfig

def process_csv_for_ipa(
    csv_path: str,
    language: str = "Chinese Mandarin",
    input_col: int = 1,
    output_suffix: str = "_with_IPA",
    config: GoruutConfig = None,
) -> bool:
    """
    Read a CSV file, convert the specified column (starting from second row) to IPA,
    and save the result with an additional IPA column.
    Server is started once and reused for all rows.
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info(f"读取 CSV 文件：{csv_path}")

        # Validate CSV structure
        if df.shape[1] <= input_col:
            logger.error(f"CSV 文件列数不足，至少需要 {input_col + 1} 列")
            return False

        # Initialize IPA column
        df["IPA"] = ""

        # Start goruut server once
        config = config or GoruutConfig()
        with goruut_server_context(config) as server:
            # Process rows with progress bar
            for idx in tqdm(range(1, len(df)), desc="处理行"):
                text = str(df.iloc[idx, input_col])
                if text and text.strip():
                    try:
                        ipa = get_ipa_phonemes(language, text, config)
                        if ipa:
                            df.at[idx, "IPA"] = ipa
                        else:
                            logger.warning(f"第 {idx+1} 行无法获取 IPA: {text}")
                    except Exception as e:
                        logger.error(
                            f"第 {idx+1} 行处理失败: {text}, 错误: {e}"
                        )
                else:
                    logger.warning(f"第 {idx+1} 行文本为空或无效")

        # Generate output file path
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}{output_suffix}{ext}"

        # Save the updated CSV
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"已保存结果到 {output_path}")
        return True

    except FileNotFoundError:
        logger.error(f"找不到 CSV 文件：{csv_path}")
        return False
    except pd.errors.EmptyDataError:
        logger.error("CSV 文件为空")
        return False
    except Exception as e:
        logger.error(f"处理 CSV 文件时出错：{e}")
        return False


def setup_logging():
    print("设置日志记录")


def main():
    parser = argparse.ArgumentParser(
        description="将 CSV 文件的指定列中文文本转为 IPA"
    )
    parser.add_argument("--csv", required=True, help="CSV 文件路径")
    parser.add_argument(
        "--language",
        default="Chinese Mandarin",
        help="语言（默认：Chinese Mandarin）",
    )
    parser.add_argument(
        "--input-col",
        type=int,
        default=1,
        help="输入文本列索引（从 0 开始，默认 1）",
    )

    args = parser.parse_args()

    config = GoruutConfig()
    if not process_csv_for_ipa(
        args.csv, args.language, args.input_col, config=config
    ):
        exit(1)


if __name__ == "__main__":
    main()
