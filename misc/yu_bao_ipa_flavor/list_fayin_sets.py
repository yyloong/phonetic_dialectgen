import copy
import json
import rich

from rich.console import Console

rich_console = Console(width=1024)

with open("./data/2025.07.11_语保公开文字信息.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# data[i]: 地理位置, 老男单字, 青男单字, 老男词汇, 老男语法
# data[i]: dict_keys(['meta_info', 'char_info', 'char_alt_info', 'word_info', 'syntax_info'])


def get_loc_str(spot):
    return spot["meta_info"]["province"] + spot["meta_info"]["city"] + spot["meta_info"]["county"]


def find_spot(data, loc_sub_str):
    for spot in data:
        loc_str = get_loc_str(spot)
        if loc_sub_str in loc_str:
            return copy.deepcopy(spot)
    return None


# def get_fayin_sets(spot):
# shengmu_set = {}
# yunmu_set = {}
# shengdiao_set = {}
# for fayin_list in spot["char_info"].values():
#     for fayin in fayin_list:
#         shengmu, yunmu, shengdiao, _ = fayin
#         shengmu_set[shengmu] = shengmu_set.get(shengmu, set()) | {shengmu + yunmu}
#         yunmu_set[yunmu] = yunmu_set.get(yunmu, set()) | {shengmu + yunmu}
#         shengdiao_set[shengdiao] = shengdiao_set.get(shengdiao, 0) + 1
# return shengmu_set, yunmu_set, shengdiao_set


def get_fayin_sets(spot):
    shengmu_set = set()
    yunmu_set = set()
    shengdiao_set = set()
    for fayin_list in spot["char_info"].values():
        for fayin in fayin_list:
            shengmu, yunmu, shengdiao, _ = fayin
            shengmu_set.add(shengmu)
            yunmu_set.add(yunmu)
            shengdiao_set.add(shengdiao)
    return shengmu_set, yunmu_set, shengdiao_set


def print_fayin_sets(loc_sub_str):
    spot = find_spot(data, loc_sub_str)
    assert not spot is None, f"未找到 {repr(loc_sub_str)}"
    shengmu_set, yunmu_set, shengdiao_set = get_fayin_sets(spot)
    rich_console.print("[bold red]meta_info[/bold red]", spot["meta_info"])
    rich_console.print("[bold red]声母[/bold red]", sorted(shengmu_set))
    rich_console.print("[bold red]韵母[/bold red]", sorted(yunmu_set))
    rich_console.print("[bold red]声调[/bold red]", sorted(shengdiao_set))


loc_sub_str = "西城"
print_fayin_sets(loc_sub_str)

loc_sub_str = "芝罘"
print_fayin_sets(loc_sub_str)

loc_sub_str = "香港"
print_fayin_sets(loc_sub_str)

"""
meta_info
{'longitude': 116.39934357, 'latitude': 39.887746569, 'province': '北京', 'city': '（无）', 'county': '西城'}
声母
['f', 'k', 'kh', 'l', 'm', 'n', 'p', 'ph', 's', 't', 'th', 'ts', 'tsh', 'tɕ', 'tɕh', 'tʂ', 'tʂh', 'x', 'Ǿ', 'ɕ', 'ʂ', 'ʐ']
韵母
['a', 'ai', 'an', 'au', 'aŋ', 'ei', 'i', 'ia', 'ian', 'iau', 'iaŋ', 'ie', 'in', 'iou', 'iuŋ', 'iŋ', 'o', 'ou', 'u', 'ua', 'uai', 'uan', 'uaŋ', 'uei', 'uo', 'uŋ', 'uən', 'uəŋ', 'y', 'yan', 'ye', 'yn', 'ən', 'ər', 'əŋ', 'ɤ', 'ɿ', 'ʅ']
声调
['214', '35', '51', '55']
meta_info
{'longitude': 121.34166666666667, 'latitude': 37.516666666666666, 'province': '山东', 'city': '烟台', 'county': '芝罘区'}
声母
['f', 'k', 'kh', 'l', 'm', 'n', 'p', 'ph', 's', 't', 'th', 'ts', 'tsh', 'tɕ', 'tɕh', 'x', 'Ǿ', 'ȵ', 'ɕ']
韵母
['a', 'an', 'au', 'aɛ', 'ei', 'i', 'ia', 'ian', 'iau', 'iaɛ', 'ie', 'in', 'iou', 'iŋ', 'iɑŋ', 'o', 'ou', 'u', 'ua', 'uan', 'uaɛ', 'uei', 'uo', 'uŋ', 'uɑŋ', 'uən', 'y', 'yan', 'yn', 'yŋ', 'yɛ', 'ɑŋ', 'ən', 'ər', 'əŋ', 'ɤ', 'ɿ']
声调
['214', '241', '33', '53']
meta_info
{'longitude': 114.175564521304, 'latitude': 22.4538143304, 'province': '香港', 'city': '（无）', 'county': '（无）'}
声母
['f', 'h', 'k', 'kh', 'l', 'm', 'n', 'p', 'ph', 's', 't', 'th', 'ts', 'tsh', 'ŋ', 'Ǿ']
韵母
['a', 'ai', 'ak', 'am', 'an', 'ap', 'at', 'au', 'aŋ', 'ei', 'ek', 'eŋ', 'i', 'iek', 'ieŋ', 'im', 'in', 'iok', 'ioŋ', 'ip', 'it', 'iu', 'iøn', 'iœk', 'iœŋ', 'iɐm', 'iɐn', 'iɐp', 'iɐt', 'iɐu', 'iɛ', 'iɛŋ', 'ok', 'ou', 'oŋ', 'u', 'ua', 'uai', 'uak', 'uan', 'uat', 'uaŋ', 'ueŋ', 'ui', 'un', 'ut', 'uɐi', 'uɐn', 'uɐt', 'uɔ', 'uɔk', 'uɔŋ', 'y', 'yn', 'yt', 'øn', 'øt', 'øy', 'ŋ', 'œ', 'œk', 'œŋ', 'ɐi', 'ɐk', 'ɐm', 'ɐn', 'ɐp', 'ɐt', 'ɐu', 'ɐŋ', 'ɔ', 'ɔi', 'ɔk', 'ɔn', 'ɔt', 'ɔŋ', 'ɛ', 'ɛk', 'ɛŋ']
声调
['13', '2', '21', '22', '3', '33', '35', '5', '55']

TODO:
    (1) 对立:
        北京: puo phuo muo fuo ai uai iuŋ uəŋ
        烟台: po pho mo fo ae uae yŋ uŋ
    (2) 烟台简化:
        ɑ -> a  # 与北京相同
        ȵ -> n  # 与北京相同
        ɛ -> e  # 与北京相同; 虽然香港有 ɛ
        c -> k  # 烟台团音
        ch -> kh  # 烟台团音
        ç -> x  # 烟台团音
"""
