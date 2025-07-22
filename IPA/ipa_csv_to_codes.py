import json
import os
import csv
import argparse
import sys

# 定义中文标点符号集合
CHINESE_PUNCTUATION = "。！？，、；：“”‘’（）【】《》"


def load_mapping_dict():
    """加载映射字典"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_file = os.path.join(script_dir, "mapping_dict.json")

    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 映射文件未找到: {mapping_file}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 映射文件格式错误: {mapping_file}")
        return None


def convert_sentence_to_codes(sentence, mapping_dict):
    """将句子转换为编码元组"""
    codes = []  # 存储编码结果

    # 遍历每个字符
    for char in sentence:
        # 尝试在字典中查找字符
        if char in mapping_dict:
            codes.append(mapping_dict[char])
        else:
            # 检查是否为中文标点
            if char in CHINESE_PUNCTUATION:
                continue  # 忽略标点
            else:
                return ()  # 非标点字符缺失映射

    return tuple(codes)


def process_csv_file(input_file, mapping_dict):
    """处理CSV文件"""
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return False

    # 准备输出文件路径
    file_dir, file_name = os.path.split(input_file)
    file_base, file_ext = os.path.splitext(file_name)
    output_file = os.path.join(file_dir, f"{file_base}_2num{file_ext}")

    # 读取并处理CSV文件
    processed_count = 0
    error_count = 0

    try:
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8", newline=""
        ) as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 处理标题行
            headers = next(reader)
            if len(headers) < 3:
                print("错误: CSV文件必须至少包含三列")
                return False

            # 写入新标题行（只保留前三列）
            writer.writerow(headers[:3])

            # 处理数据行
            for row in reader:
                # 确保至少有前三列
                if len(row) < 3:
                    error_count += 1
                    continue

                # 只保留前三列
                audio, text, ipa = row[:3]
                ipa = ipa.strip()

                # 转换IPA句子
                codes = convert_sentence_to_codes(ipa, mapping_dict)

                # 写入新行（只保留前三列）
                if codes:
                    writer.writerow([audio, text, str(codes)])
                    processed_count += 1
                else:
                    # 转换失败时记录错误
                    error_count += 1
                    print(f"警告: 转换失败 - {audio}: {ipa}")

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

    print(f"处理完成: {processed_count} 行成功, {error_count} 行失败")
    print(f"结果已保存至: {output_file}")
    return True


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(
        description="将CSV文件中的IPA列转换为编码元组"
    )
    parser.add_argument("files", nargs="+", help="要处理的CSV文件路径")
    args = parser.parse_args()

    # 加载映射字典
    mapping_dict = load_mapping_dict()
    if mapping_dict is None:
        print("无法加载映射字典，程序退出。")
        sys.exit(1)

    # 处理每个文件
    for file_path in args.files:
        print(f"\n处理文件: {file_path}")
        if not process_csv_file(file_path, mapping_dict):
            print(f"文件处理失败: {file_path}")


if __name__ == "__main__":
    main()
