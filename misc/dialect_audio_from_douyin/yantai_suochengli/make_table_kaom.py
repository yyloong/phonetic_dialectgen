"""
HTML表格格式转换器（参考 kaom_extract.py 逻辑重写）
=====================================================

使用 BeautifulSoup 并参考用户提供的正确代码逻辑重新编写，确保：
1. 正确处理HTML表格结构 (rowspan)
2. 正确解析声调和汉字
3. 正确生成与 qian1981 对照的表格

本脚本用于从 kaom.net 网站的 HTML 文件中提取烟台方言语音数据，
并将其转换为标准化的 HTML 表格格式，便于与其他数据源进行对比分析。

数据来源：http://www.kaom.net/si_x8.php?c=C139
处理逻辑：完全参考 misc/kaom_extract.py 中的 extract_raw 函数
"""

import re
from collections import defaultdict
from bs4 import BeautifulSoup

from opencc import OpenCC

cc_t2s = OpenCC("t2s")


def extract_raw_data_from_kaom(soup: BeautifulSoup):
    """
    从 kaom.net 的 HTML 中提取原始语音数据。
    逻辑完全参考 misc/kaom_extract.py 中的 extract_raw。

    参数:
        soup (BeautifulSoup): 解析后的 HTML 文档对象

    返回:
        list: 包含 (声母, 韵母, 声调汉字) 三元组的列表

    功能说明:
        1. 定位到 kuangjia div 中的主要数据表格
        2. 逐行处理表格数据，正确处理 rowspan 属性
        3. 提取声调标记（<i>标签）和对应汉字
        4. 组织数据为 (声母, 韵母, 声调汉字) 格式
    """
    # 定位到包含语音数据的主表格
    # kaom.net 的页面结构：<div class="kuangjia"> 包含主要内容
    table = soup.find("div", class_="kuangjia").find("table")
    rows = table.find_all("tr")

    # 用于存储解析后的表格矩阵
    matrix = []

    # 用于跟踪跨行单元格（rowspan）的位置和内容
    # 格式：{(行索引, 列索引): 单元格内容}
    rowspan_tracker = {}

    # 逐行处理表格数据
    for row_idx, row in enumerate(rows):
        # 提取当前行的所有单元格（包括 th 和 td）
        cols = row.find_all(["td", "th"])
        row_data = []
        col_idx = 0

        # 处理由于 rowspan 导致的"虚拟"单元格
        # 当前行可能因为上一行的 rowspan 而需要插入额外的单元格
        while col_idx < len(matrix[0]) if matrix else 0:
            if (row_idx, col_idx) in rowspan_tracker:
                # 从跨行跟踪器中获取内容并插入当前行
                text = rowspan_tracker.pop((row_idx, col_idx))
                row_data.append(text)
                col_idx += 1
            else:
                break

        # 处理当前行中实际存在的单元格
        for cell in cols:
            # 继续处理可能的 rowspan 单元格
            while (row_idx, col_idx) in rowspan_tracker:
                text = rowspan_tracker.pop((row_idx, col_idx))
                row_data.append(text)
                col_idx += 1

            # 提取单元格文本内容
            text = ""

            # 判断单元格类型和处理方式
            if row_idx <= 0 or col_idx <= 2:
                # 表头行或前三列（韻尾、介音、韻母）：直接提取纯文本
                text += cell.get_text(strip=True)
            elif cell.get_text(strip=True) == "":
                # 空单元格：不做处理
                pass
            else:
                # 数据单元格：包含声调标记和汉字的复合内容
                # 需要解析 <i>标签（声调）和后续文本（汉字）

                # 遍历单元格的所有子元素
                for content in cell.contents:
                    if content.name == "i":
                        # <i>标签包含声调数值（如：213, 31, 55）
                        diao_notation = content.get_text(strip=True)
                        text += f"{diao_notation}:"
                    elif (
                        getattr(content, "name", None) is None and content.strip()
                    ):  # NavigableString
                        # 普通文本节点，包含汉字
                        characters = content.strip()
                        if text.endswith(":"):
                            # 确保声调标记后紧跟汉字
                            text += f"{characters},"

                # 去除末尾的逗号，形成最终格式：213:八把靶,31:巴疤笆,55:壩拔罷雹霸
                text = text.rstrip(",")

            # 将处理后的文本添加到当前行数据中
            row_data.append(text)

            # 处理 rowspan 属性：记录需要跨越的行
            rowspan = int(cell.get("rowspan", 1))
            if rowspan > 1:
                # 为接下来的 rowspan-1 行在相同列位置记录相同内容
                for r in range(1, rowspan):
                    rowspan_tracker[(row_idx + r, col_idx)] = text
            col_idx += 1

        # 处理行尾可能剩余的 rowspan 单元格
        while (row_idx, col_idx) in rowspan_tracker:
            text = rowspan_tracker.pop((row_idx, col_idx))
            row_data.append(text)
            col_idx += 1

        # 将完整的行数据添加到矩阵中
        matrix.append(row_data)

    # 验证表格结构是否符合预期
    if matrix[0][0] != "韻尾" or matrix[0][1] != "介音" or matrix[0][2] != "韻母":
        raise ValueError("表格结构不符合预期")

    # 去除前两列（韻尾、介音），只保留韻母列和声母列
    # 这是因为我们只关心韻母和声母的组合
    matrix = [row[2:] for row in matrix]

    # 将二维矩阵转换为 (声母, 韵母, 声调汉字) 三元组列表
    data = []
    header = matrix[0]  # 表头行：第一个是韻母，后面是各个声母

    # 遍历数据行（跳过表头）
    for row_idx, row in enumerate(matrix[1:], start=1):
        yun = row[0]  # 当前行的韵母

        # 遍历当前行的各个声母列
        for col_idx, text in enumerate(row[1:], start=1):
            if text:  # 只处理非空单元格
                sheng = header[col_idx]  # 对应的声母
                data.append((sheng, yun, text))

    return data


def process_tone_data(data):
    """
    处理声调数据，将格式化的字符串转换为结构化数据

    参数:
        data (list): 包含 (声母, 韵母, 声调汉字) 三元组的列表

    返回:
        defaultdict: 以音节为键，按声调顺序排列的汉字列表为值

    功能说明:
        1. 解析声调标记和汉字的组合字符串
        2. 将数据按音节分组
        3. 按标准声调顺序（31, 214, 55）重新排列
        4. 处理特殊情况如零声母
    """
    # 用于存储处理后的音节数据
    processed_data = defaultdict(list)

    # 逐个处理原始数据项
    for sheng, yun, text in data:
        # 处理零声母的特殊情况
        # 在 kaom.net 中，零声母用"零"表示，我们转换为空字符串
        if sheng == "零":
            sheng = ""

        # 组合声母和韵母形成完整音节
        syllable = sheng + yun

        # NOTE: 将213转换为214 & 繁体转换为简体
        text = text.replace("213", "214")
        text = cc_t2s.convert(text)

        # 解析声调和汉字的组合字符串
        if ":" in text:
            # 标准格式：包含声调标记，如 "213:八把靶,31:巴疤笆,55:壩拔罷雹霸"
            tone_parts = text.split(",")
            tone_dict = {}

            # 分离各个声调的数据
            for part in tone_parts:
                if ":" in part:
                    tone, chars = part.split(":", 1)
                    tone = tone.strip()
                    chars = chars.strip()
                    if tone and chars:
                        tone_dict[tone] = chars

            # 按标准声调顺序重新排列数据
            # 31: 高平调, 214: 低升调, 55: 高平调
            tone_order = ["31", "214", "55"]
            for tone_marker in tone_order:
                # 如果该声调存在数据则添加，否则添加空字符串占位
                processed_data[syllable].append(tone_dict.get(tone_marker, ""))
        else:
            # 非标准格式：直接包含汉字，无声调标记
            if text:
                processed_data[syllable].append(text)

    return processed_data


def extract_initials_finals(syllable_data):
    """
    从音节数据中提取声母和韵母列表

    参数:
        syllable_data (dict): 音节数据字典，键为音节，值为汉字列表

    返回:
        tuple: (声母列表, 韵母列表)，均为排序后的列表

    功能说明:
        1. 使用预定义的声母模式识别每个音节的声母部分
        2. 提取剩余部分作为韵母
        3. 去重并排序，用于构建表格的行列标题
    """
    # 用于收集所有出现的声母和韵母
    initials = set()
    finals = set()

    # 预定义的声母模式列表
    # 按长度降序排列，确保优先匹配较长的声母（如 tsh 而非 t+s+h）
    initial_patterns = sorted(
        [
            "tsh",
            "tɕh",
            "ph",
            "th",
            "ch",
            "kh",  # 送气声母
            "tɕ",
            "ts",
            "ȵ",
            "ɕ",
            "ç",  # 塞擦音和擦音
            "p",
            "m",
            "f",
            "t",
            "n",
            "l",
            "k",
            "s",
            "x",
            "c",  # 基本声母
        ],
        key=len,
        reverse=True,
    )

    # 逐个分析每个音节
    for syllable in syllable_data.keys():
        # 寻找匹配的声母
        initial = ""
        for pattern in initial_patterns:
            if syllable.startswith(pattern):
                initial = pattern
                break

        # 剩余部分作为韵母
        final = syllable[len(initial) :]

        # 添加到对应的集合中
        initials.add(initial)
        finals.add(final)

    # 返回排序后的列表，用于构建表格
    return sorted(list(initials)), sorted(list(finals))


def generate_html_table(syllable_data):
    """
    生成HTML表格

    参数:
        syllable_data (dict): 处理后的音节数据

    返回:
        str: 完整的HTML表格字符串

    功能说明:
        1. 提取声母和韵母列表作为表格的行列标题
        2. 创建标准化的HTML表格结构
        3. 为每个音节填充对应的声调和汉字数据
        4. 应用统一的CSS样式
    """
    # 获取所有声母和韵母，用作表格的行列标题
    initials, finals = extract_initials_finals(syllable_data)

    # 声调标记，用于显示在表格中
    tone_markers = ["31", "214", "55"]

    # 生成HTML文档的开始部分
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>烟台方言音系 kaom.net</title>
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
    <h1>烟台方言语音表 <code>http://www.kaom.net/si_x8.php?c=C139</code></h1>
    <table>
        <tr>
            <th>韵母/声母</th>
"""

    # 添加声母表头行
    # 每列对应一个声母，空声母显示为"零"
    for initial in initials:
        display_initial = initial if initial else "零"
        html += f"            <th>{display_initial}</th>\n"

    html += "        </tr>\n"

    # 添加数据行
    # 每行对应一个韵母，与各声母形成音节组合
    for final in finals:
        html += f"        <tr>\n            <th>{final}</th>\n"

        # 遍历每个声母，形成当前韵母与各声母的组合
        for initial in initials:
            syllable = initial + final
            cell_content = ""

            # 如果该音节有数据，则填充到表格单元格中
            if syllable in syllable_data:
                data = syllable_data[syllable]

                # 处理最多3个声调的数据
                for i in range(3):
                    # 获取对应声调的汉字，如果没有则为空
                    characters = data[i] if i < len(data) else ""
                    tone_marker = tone_markers[i] if i < len(tone_markers) else ""

                    # 只有当有汉字数据或者至少有一个声调有数据时才显示行
                    if characters or (i < len(tone_markers) and any(d for d in data)):
                        cell_content += f'                <div class="tone-line"><span class="tone-marker">{tone_marker}</span> {characters}</div>\n'

            # 将单元格内容添加到表格中
            html += f'            <td class="syllable-cell">\n{cell_content}            </td>\n'

        html += "        </tr>\n"

    # 结束HTML文档
    html += """    </table>
</body>
</html>"""

    return html


def main():
    """
    主函数：协调整个转换流程

    流程说明:
        1. 读取并解析 kaom.net 的 HTML 文件
        2. 提取原始的语音数据
        3. 处理和标准化声调数据
        4. 生成标准化的 HTML 表格
        5. 保存结果并输出统计信息
    """
    # 定义输入和输出文件名
    input_file = "kaom_net_si_x8s_php_C139.html"
    output_file = "table_kaom.html"

    print(f"正在解析HTML文件: {input_file}...")

    # 读取并解析HTML文件
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_file}'")
        return

    # 第一步：提取原始数据
    raw_data = extract_raw_data_from_kaom(soup)
    print(f"提取到 {len(raw_data)} 条原始数据")

    # 第二步：处理声调数据
    processed_data = process_tone_data(raw_data)
    print(f"处理得到 {len(processed_data)} 个音节")

    # 第三步：生成HTML表格
    html_output = generate_html_table(processed_data)

    # 第四步：保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"转换完成！输出文件：{output_file}")

    # 输出统计信息
    initials, finals = extract_initials_finals(processed_data)
    print(f"声母: {len(initials)}个, 韵母: {len(finals)}个")


if __name__ == "__main__":
    main()
