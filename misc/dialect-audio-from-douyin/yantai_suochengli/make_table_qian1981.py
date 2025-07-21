import re
from collections import defaultdict


def parse_dialect_file(filename):
    """
    解析方言数据文件，提取音节数据

    参数:
        filename (str): 原始方言数据文件路径

    返回:
        defaultdict: 以音节为键，汉字列表为值的字典

    功能说明:
        - 读取原始文本文件中的每一行
        - 解析每行的音节和对应汉字
        - 将同一音节的不同声调汉字分组存储
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 用于存储音节数据的字典，键为音节，值为汉字列表
    syllable_data = defaultdict(list)

    # 逐行处理文本数据
    for line in lines:
        line = line.strip()  # 去除首尾空白字符
        if not line:  # 跳过空行
            continue

        # 按第一个空格分割，获取音节和汉字部分
        # 格式如："pa 巴芭疤" -> ["pa", "巴芭疤"]
        parts = line.split(" ", 1)
        if len(parts) < 2 or parts[1].strip() == "":
            parts = [parts[0], "-"]

        syllable = parts[0]  # 音节部分（如："pa"）
        characters = parts[1] if len(parts) > 1 else ""  # 汉字部分（如："巴芭疤"）

        # 将汉字添加到对应音节的列表中
        # 同一音节的不同声调会形成多个条目
        syllable_data[syllable].append(characters)

    return syllable_data


def extract_initials_finals(syllable_data):
    """
    从音节数据中提取声母和韵母

    参数:
        syllable_data (dict): 音节数据字典

    返回:
        tuple: (声母列表, 韵母列表) 均为排序后的列表

    功能说明:
        - 根据预定义的声母模式识别每个音节的声母
        - 提取剩余部分作为韵母
        - 返回所有唯一的声母和韵母，用于构建表格
    """
    initials = set()  # 存储所有声母
    finals = set()  # 存储所有韵母

    # 常见声母模式列表，按长度降序排列以优先匹配长声母
    # 例如："tsh" 应该优先于 "ts"，"t" 和 "s" 匹配
    initial_patterns = [
        "ph",
        "th",
        "tsh",
        "ch",
        "kh",
        "tɕh",
        "tɕ",
        "ts",
        "p",
        "m",
        "f",
        "t",
        "n",
        "l",
        "ȵ",
        "ɕ",
        "c",
        "k",
        "ç",
        "x",
        "s",
    ]

    # 遍历所有音节，分离声母和韵母
    for syllable in syllable_data.keys():
        # 寻找匹配的声母
        initial = ""
        for pattern in sorted(initial_patterns, key=len, reverse=True):
            if syllable.startswith(pattern):
                initial = pattern
                break

        # 剩余部分作为韵母
        final = syllable[len(initial) :]

        # 添加到对应的集合中
        initials.add(initial)
        finals.add(final)

    # 返回排序后的声母和韵母列表
    return sorted(initials), sorted(finals)


def generate_table(syllable_data):
    """
    生成HTML格式的语音表

    参数:
        syllable_data (dict): 音节数据字典

    返回:
        str: 完整的HTML表格字符串

    功能说明:
        - 提取声母和韵母信息
        - 创建HTML表格结构
        - 用声调符号替换原音节标记
        - 生成美观的样式表
    """

    # 提取声母和韵母信息
    initials, finals = extract_initials_finals(syllable_data)

    # 声调标记符号，用于替换原音节标记
    # 31: 平声, 214: 上声, 55: 去声
    tone_markers = ["31", "214", "55"]

    # Start HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>烟台方言音系 钱曾怡1981</title>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .syllable-cell {
            font-size: 12px;
            line-height: 1.4;
        }
        .tone-line {
            margin: 2px 0;
        }
        .tone-marker {
            font-weight: bold;
            color: #0066cc;
        }
    </style>
</head>
<body>
    <h1>烟台方言语音表 《烟台方言调查报告》 钱曾怡 等 1981</h1>
    <table>
        <tr>
            <th>韵母/声母</th>
"""

    # 添加表头行，包含所有声母（辅音）
    for initial in initials:
        html += f"            <th>{initial}</th>\n"

    html += "        </tr>\n"

    # 添加数据行，每行代表一个韵母（元音）
    for final in finals:
        html += f"        <tr>\n            <th>{final}</th>\n"

        # 遍历每个声母，形成该韵母与各声母的组合
        for initial in initials:
            syllable = initial + final
            cell_content = ""

            if syllable in syllable_data:
                data = syllable_data[syllable]

                # 处理最多3个条目（对应3个声调）
                for i in range(3):
                    if i < len(data):
                        characters = data[i]
                    else:
                        characters = ""

                    tone_marker = tone_markers[i] if i < len(tone_markers) else ""

                    if characters or tone_marker:
                        cell_content += f'                <div class="tone-line"><span class="tone-marker">{tone_marker}</span> {characters}</div>\n'

            html += f'            <td class="syllable-cell">\n{cell_content}            </td>\n'

        html += "        </tr>\n"

    html += """    </table>
</body>
</html>"""

    return html


def main():
    # Parse the dialect file
    syllable_data = parse_dialect_file("qian1981_raw.txt")

    # Generate the HTML table
    html_table = generate_table(syllable_data)

    # Save to file
    with open("table_qian1981.html", "w", encoding="utf-8") as f:
        f.write(html_table)

    print("表格已生成: table_qian1981.html")
    print(f"共处理了 {len(syllable_data)} 个音节")


if __name__ == "__main__":
    main()
