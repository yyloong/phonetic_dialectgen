import os
import re
import multiprocessing as mp

from tqdm import tqdm

from bs4 import BeautifulSoup

"""
几种数据情形

(1) 实际调值

C139.html: 213八把靶31巴疤笆55壩拔罷雹霸

(2) 实际调值及其四声对应类型

Z008.html: 15-去聲把壩罷霸垻■23-陽平爸八跋41-上聲把靶■45-陰平吧巴疤芭

(3) 实际调值 IPA Unicode ( https://zh.wikipedia.org/wiki/%E4%BA%94%E5%BA%A6%E6%A0%87%E8%AE%B0%E6%B3%95 )

Z007.html: ˥˩杷耙˦˥垻壩弝欛灞爸耙霸靶˦˦叭吧巴爸疤笆粑芭豝鈀˧˧把˨˨爸笆˩˧˩巴扒掱杷爬琶筢耙

(4) 四声

B075.html: 1巴疤芭3八剝叭把5壩把拔爸罷雹霸

(5) 四声

Z010.html: [1]爸[3]擺[5]拜[7]百柏伯迫不

(6) 日本音读

Z103.html: 零打

(7) 朝鲜音读

Z106.html: Ⓛ琶波婆零巴

(8) 越南音读

Z107.html: [問]把跁跛[平]巴笆羓豝爬鈀筢芭葩疤波番[玄]婆琶杷番皤[重]簿簿[銳]伯播欛百霸百佰柏栢迫
"""

# 以下代码仅考虑情形 (1), (2), (3)


def _load_unicodes_diao_dict():
    """
    从文件中加载 Unicode 声调符号到数字声调值的映射字典。

    此函数读取名为 'unicodes_diao_to_numbers.txt' 的文本文件，
    该文件的每一行包含一个 Unicode 声调符号和对应的数字声调值，
    两者之间用空格分隔。函数将这些信息存储在一个字典中并返回。

    Returns:
        dict: 一个将 Unicode 声调符号映射到数字声调值的字典。
    """
    unicodes_diao = {}
    with open("unicodes_diao_to_numbers.txt", mode="rt", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            unicodes, diao = line.strip().split(" ")
            unicodes_diao[unicodes] = diao
    return unicodes_diao


def _compile_pattern_abundant_info():
    """
    编译一个正则表达式模式，用于匹配冗余的声调信息，如声调名称和分隔符。

    此函数构建一个包含多种声调相关信息的正则表达式，
    包括声调名称（如平、上、去、入）及其阴阳分类，以及未知汉字占位符（■）。

    Returns:
        re.Pattern: 一个编译好的正则表达式模式对象。
    """
    abundant_infos = ["■"]
    for c in "平上去入":
        # [!] 正则表达式按顺序匹配, 从长到短倒叙排列!
        abundant_infos.append(r"\-陰" + c)
        abundant_infos.append(r"\-陽" + c)
        abundant_infos.append(r"\-" + c + "聲")
        abundant_infos.append(r"\-" + c)
    abundant_infos.append(r"\-舒入")
    regex = "|".join(abundant_infos)
    pattern = re.compile(regex)
    return pattern


def _compile_pattern_unicodes_diao(unicodes_diao):
    """
    编译一个正则表达式模式，用于匹配 Unicode 声调符号。

    此函数根据传入的 Unicode 声调符号到数字声调值的映射字典，
    构建一个正则表达式模式，用于匹配字典中的所有 Unicode 声调符号。

    Args:
        unicodes_diao (dict): 一个将 Unicode 声调符号映射到数字声调值的字典。

    Returns:
        re.Pattern: 一个编译好的正则表达式模式对象。
    """
    # [!] 正则表达式按顺序匹配, 从长到短倒叙排列!
    regex = "|".join(sorted(unicodes_diao.keys(), key=len, reverse=True))
    pattern = re.compile(regex)
    return pattern


UNICODES_DIAO = _load_unicodes_diao_dict()

PATTERN_CONSECUTIVE_DIGITS = re.compile(r"\d+")
PATTERN_DUPLICATED_DIGITS = re.compile(r"(\d)\1+")
PATTERN_SINGLE_DIGIT = re.compile(r"(?<!\d)(\d)(?!\d)")
PATTERN_ABUNDANT_INFO = _compile_pattern_abundant_info()
PATTERN_UNICODES_DIAO = _compile_pattern_unicodes_diao(UNICODES_DIAO)


# 排除 (4), (5) 仅有四声不注明实际调值的情形
def filter_notation(data):
    """
    过滤数据，排除仅有四声但不注明实际调值的情形。

    此函数会对输入的数据进行多步处理，包括去除冗余的声调信息、替换 Unicode 声调符号为数字声调值。

    Args:
        data (list): 包含声调信息的元组列表，每个元组格式为 (sheng, yun, text)。
    Returns:
        list: 经过过滤和处理后的数据列表；如果数据不符合要求，返回 None。
    """
    original_data = data
    cat_text = "".join([text for _, _, text in data])
    consecutive_digits_matchs = PATTERN_CONSECUTIVE_DIGITS.findall(cat_text)
    if len(consecutive_digits_matchs) > 0 and max(map(len, consecutive_digits_matchs)) <= 1:
        return None
    clean_data = []
    for sheng, yun, text in data:
        text = PATTERN_ABUNDANT_INFO.sub("", text)
        clean_data.append((sheng, yun, text))
    data = clean_data
    clean_data = []
    if any(map(lambda u: u in cat_text, "˩˨˧˦˥")):
        for sheng, yun, text in data:
            text = PATTERN_UNICODES_DIAO.sub(lambda m: UNICODES_DIAO.get(m.group(0), "~"), text)
            text = PATTERN_DUPLICATED_DIGITS.sub(r"\1", text)
            # 非入声的音高不变的情形需要恢复
            if not any(map(lambda ru: yun.endswith(ru), "ptkʔ")):
                text = PATTERN_SINGLE_DIGIT.sub(lambda m: m.group(1) * 2, text)
            clean_data.append((sheng, yun, text))
        data = clean_data
        clean_data = []
    return data


def extract_diao(data3):
    """
    从输入的数据中提取声调信息，并将其拆分为更详细的格式。

    此函数接收一个包含声调信息的元组列表，每个元组格式为 (sheng, yun, text)。
    它会将 text 中的内容按逗号分隔，再按冒号分割出声调值和对应的字符，
    最终将结果整理成新的元组列表，每个元组格式为 (sheng, yun, diao, characters)。

    Args:
        data3 (list): 包含声调信息的元组列表，每个元组格式为 (sheng, yun, text)。

    Returns:
        list: 经过处理后的数据列表，每个元组格式为 (sheng, yun, diao, characters)。
    """
    data4 = []
    for sheng, yun, text in data3:
        for t in text.split(","):
            diao, characters = t.split(":")
            data4.append((sheng, yun, diao, characters))
    return data4


# 排除 (6), (7), (8) 非汉语的情形
def extract_language(soup: BeautifulSoup):
    """
    从 HTML 文档中提取语言信息，并排除非汉语或未知的情形。

    此函数在 HTML 文档中查找特定的 <div> 元素，从中提取语言信息。
    如果语言信息包含 '未知' 或 '域外音'，则不再做后续处理，返回 None。

    Args:
        soup (BeautifulSoup): 一个 BeautifulSoup 对象，代表解析后的 HTML 文档。

    Returns:
        str or None: 如果语言信息为已知采集地点的汉语方言，返回语言名称；否则返回 None。
    """
    div = soup.find("div", style=lambda s: s and "margin: 20px auto 10px auto" in s)
    if not div:
        return None
    text = div.get_text(strip=True)
    language = text.split("語言點：")[1].split("　字數：")[0]
    if "未知" in language or "域外音" in language:
        return None
    return language


def extract_raw(soup: BeautifulSoup):
    table = soup.find("div", class_="kuangjia").find("table")
    rows = table.find_all("tr")

    matrix = []
    rowspan_tracker = {}

    for row_idx, row in enumerate(rows):
        cols = row.find_all(["td", "th"])
        row_data = []
        col_idx = 0
        rowspan_shift = 0

        while col_idx - rowspan_shift < len(cols):

            # [!] 教训: rowspan_shift, 思考 tbody 元素的结构!
            # [!] 教训: 开头必须 `cell = cols[col_idx]`, 结尾必须 `col_idx += 1`, 两句必须成对出现!

            while (row_idx, col_idx) in rowspan_tracker:
                cell = cols[col_idx]
                text = rowspan_tracker.pop((row_idx, col_idx))
                row_data.append(text)
                col_idx += 1
                rowspan_shift += 1

            cell = cols[col_idx - rowspan_shift]

            text = ""
            if row_idx <= 0 or col_idx <= 2:
                text += cell.get_text(strip=True)
            elif cell.get_text(strip=True) == "":
                pass
            else:
                for content in cell.contents:
                    if content.name == "i":
                        diao_notation = content.get_text(strip=True)
                        text += f"{diao_notation}:"
                    elif content.text:
                        assert text.endswith(":")
                        characters = content.strip()
                        text += f"{characters},"
                text = text.rstrip(",")
            row_data.append(text)

            rowspan = int(cell.get("rowspan", 1))
            for r in range(1, rowspan):
                rowspan_tracker[(row_idx + r, col_idx)] = text

            col_idx += 1

        if len(matrix) > 1:
            assert len(row_data) == len(matrix[0])
        matrix.append(row_data)

    assert matrix[0][0] == "韻尾"
    assert matrix[0][1] == "介音"
    assert matrix[0][2] == "韻母"

    # 去除前两列, 韻尾, 介音
    matrix = [row[2:] for row in matrix]

    data = []
    for row_idx, row in enumerate(matrix):
        for col_idx, text in enumerate(row):
            if row_idx == 0 or col_idx == 0:
                continue
            if text == "":
                continue
            sheng = matrix[0][col_idx]
            yun = matrix[row_idx][0]
            data.append((sheng, yun, text))

    return data


def process_html(html_path):
    with open(html_path, mode="rt", encoding="utf-8") as file:
        html = file.read()
        soup = BeautifulSoup(html, "html.parser")
        language = extract_language(soup)
        if not language:
            return None
        data = extract_raw(soup)
        data = filter_notation(data)
        if data is None:
            return None
        data = extract_diao(data)
        return language, sorted(data)


if __name__ == "__main__":

    task_list = []
    html_dir = "./kaom_raw/data"
    html_files = os.listdir(html_dir)
    # random.shuffle(html_files)
    for html_file in html_files:
        if html_file.endswith(".html"):
            html_path = os.path.join(html_dir, html_file)
            task_list.append(html_path)

    all_phonologies = {}
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.imap_unordered(process_html, task_list)
        for result in tqdm(results, total=len(task_list), ncols=80):
            if result is None:
                continue
            language, data = result
            all_phonologies[language] = data

    all_phonologies = sorted(all_phonologies.items())

    with open("kaom_clean/all_phonologies.0.json", mode="wt", encoding="utf-8") as f:
        f.write("{\n")
        for language, data in all_phonologies:
            f.write(f'  "{language}": [\n')
            for sheng, yun, diao, characters in data:
                f.write(f'    ["{sheng}", "{yun}", "{diao}", "{characters}"],\n')
            f.write(f"  ],\n")
        f.write("}\n")
