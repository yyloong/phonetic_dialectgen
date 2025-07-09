import pandas as pd
import argparse
import os
import logging
from tqdm import tqdm
from goruut_ipa import get_ipa_phonemes, goruut_server_context, GoruutConfig


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


def process_csv_for_ipa(csv_path, language, input_col, config=None):
    if not os.path.exists(csv_path):
        logging.error(f"CSV 文件不存在: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"读取 CSV 文件失败: {e}")
        return False

    if input_col >= len(df.columns):
        logging.error(f"输入列索引 {input_col} 超出范围")
        return False

    input_col_name = df.columns[input_col]
    output_col_name = f"{input_col_name}_ipa"

    with goruut_server_context(config=config) as client:
        tqdm.pandas(desc="转换为 IPA")
        df[output_col_name] = df[input_col_name].progress_apply(
            lambda x: get_ipa_phonemes(x, language=language, client=client)
        )

    output_csv_path = csv_path.replace(".csv", "_ipa.csv")
    df.to_csv(output_csv_path, index=False)
    logging.info(f"转换完成，结果已保存到: {output_csv_path}")

    return True
