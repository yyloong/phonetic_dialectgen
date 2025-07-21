import csv
import re
import sys
import os


def clean_text(text):
    """处理文本：1.移除括号内容 2.移除所有引号"""
    # 移除括号及其内部内容（包括中文/英文括号）
    text = re.sub(r"[\(（].*?[\)）]", "", text)
    # 移除所有类型的引号（单引号、双引号、中文引号）
    text = re.sub(r"[\'\"“”‘’]", "", text)
    return text.strip()


def process_csv(input_file):
    """处理CSV文件并生成新的CSV"""
    output_file = os.path.join(os.path.dirname(input_file), "data.csv")

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row["sentence"] = clean_text(row["sentence"])
            writer.writerow(row)

    print(f"处理完成！新文件已保存至: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("请提供CSV文件路径作为参数")
        print("示例: python script.py /path/to/your/file.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"错误: 文件不存在 - {input_path}")
        sys.exit(1)

    process_csv(input_path)
