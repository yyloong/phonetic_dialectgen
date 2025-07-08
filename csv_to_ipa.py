import pandas as pd
import argparse
import os
import logging
from tqdm import tqdm
from goruut_ipa import get_ipa_phonemes, goruut_server_context, GoruutConfig

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
