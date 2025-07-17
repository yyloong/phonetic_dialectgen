"""
This file is copied from https://github.com/stefantaubert/pinyin-to-ipa
and modified to fit our needs.
这个文件来自 https://github.com/stefantaubert/pinyin-to-ipa
并修改了部分代码以适应我们的需求。
"""

import sys
import itertools
from typing import Dict, Generator, List, Optional, Tuple

from ordered_set import OrderedSet
from pypinyin.contrib.tone_convert import to_finals, to_initials, to_normal, to_tone3


# 语保 北京-北京-西城
INITIAL_MAPPING: Dict[str, List[Tuple[str, ...]]] = {
    "b": [("p",)],
    "c": [("tsh",)],
    "ch": [("tʂh",)],
    "d": [("t",)],
    "f": [("f",)],
    "g": [("k",)],
    "h": [("x",)],
    "j": [("tɕ",)],
    "k": [("kh",)],
    "l": [("l",)],
    "m": [("m",)],
    "n": [("n",)],
    "p": [("ph",)],
    "q": [("tɕh",)],
    "r": [("ʐ",)],
    "s": [("s",)],
    "sh": [("ʂ",)],
    "t": [("th",)],
    "x": [("ɕ",)],
    "z": [("ts",)],
    "zh": [("tʂ",)],
}

__our_flavor = set()
for __shengmu_list in INITIAL_MAPPING.values():
    for __shengmu in __shengmu_list:
        __our_flavor.add("".join(__shengmu))
__yubao_xicheng = eval(
    "{'tʂ', 'ʂ', 't', 'l', 'th', 'ʐ', 'tɕ', 'n', 'Ǿ', 'x', 'ts', 's', 'kh', 'ɕ', 'f', 'ph', 'm', 'tsh', 'tʂh', 'k', 'tɕh', 'p'}"
)
assert len(__our_flavor - __yubao_xicheng) == 0, f"额外包含: {repr(__our_flavor - __yubao_xicheng)}"
print(f"未包含: {repr(__yubao_xicheng - __our_flavor)}", file=sys.stderr)

INITIALS = INITIAL_MAPPING.keys()

# 语保 北京-北京-西城
SYLLABIC_CONSONANT_MAPPINGS: Dict[str, List[Tuple[str, ...]]] = {
    "hm": [("h", "m", "#")],
    "hng": [("h", "ŋ", "#")],
    "m": [("m", "#")],
    "n": [("n", "#")],
    "ng": [("ŋ", "#")],
}

SYLLABIC_CONSONANTS = SYLLABIC_CONSONANT_MAPPINGS.keys()

# 语保 北京-北京-西城
INTERJECTION_MAPPINGS: Dict[str, List[Tuple[str, ...]]] = {
    "o": [("o", "#")],
    "er": [("ər", "#")],
}

INTERJECTIONS = INTERJECTION_MAPPINGS.keys()


FINAL_MAPPING: Dict[str, List[Tuple[str, ...]]] = {
    "a": [("a", "#")],
    "ai": [("ai", "#")],
    "an": [("a", "n", "#")],
    "ang": [("a", "ŋ", "#")],
    "ao": [("au", "#")],
    "e": [("ɤ", "#")],
    "ei": [("ei", "#")],
    "en": [("ə", "n", "#")],
    "eng": [("ə", "ŋ", "#")],
    "i": [("i", "#")],
    "ia": [("i", "a", "#")],
    "ian": [("i", "a", "n", "#")],
    "iang": [("i", "a", "ŋ", "#")],
    "iao": [("i", "au", "#")],
    "ie": [("i", "e", "#")],
    "in": [("i", "n", "#")],
    "iou": [("i", "ou", "#")],
    "ing": [("i", "ŋ", "#")],
    "iong": [("i", "u", "ŋ", "#")],
    "ong": [("u", "ŋ", "#")],
    "ou": [("ou", "#")],
    "u": [("u", "#")],
    "uei": [("u", "ei", "#")],
    "ua": [("u", "a", "#")],
    "uai": [("u", "ai", "#")],
    "uan": [("u", "a", "n", "#")],
    "uen": [("u", "ə", "n", "#")],
    "uang": [("u", "a", "ŋ", "#")],
    "ueng": [("u", "ə", "ŋ", "#")],
    "uo": [("u", "o", "#")],
    "o": [("u", "o", "#")],
    "ü": [("y", "#")],
    "üe": [("y", "e", "#")],
    "üan": [("y", "a", "n", "#")],
    "ün": [("y", "n", "#")],
}

FINALS = FINAL_MAPPING.keys()
__our_flavor = set()
for __yunmu_list in FINAL_MAPPING.values():
    for __yunmu in __yunmu_list:
        __our_flavor.add("".join(__yunmu).rstrip("#"))
__yubao_xicheng = eval(
    "{'uəŋ', 'əŋ', 'uei', 'ɿ', 'ia', 'uo', 'yn', 'yan', 'ou', 'a', 'ər', 'ʅ', 'iou', 'y', 'u', 'iau', 'i', 'ei', 'iuŋ', 'ye', 'o', 'au', 'iaŋ', 'ua', 'aŋ', 'uai', 'ie', 'in', 'ai', 'uan', 'iŋ', 'uən', 'ɤ', 'an', 'ian', 'ən', 'uaŋ', 'uŋ'}"
)
assert len(__our_flavor - __yubao_xicheng) == 0, f"额外包含: {repr(__our_flavor - __yubao_xicheng)}"
print(f"未包含: {repr(__yubao_xicheng - __our_flavor)}", file=sys.stderr)

# 语保 北京-北京-西城
FINAL_MAPPING_AFTER_ZH_CH_SH_R: Dict[str, List[Tuple[str, ...]]] = {
    "i": [("ʅ", "#")],
}

# 语保 北京-北京-西城
FINAL_MAPPING_AFTER_Z_C_S: Dict[str, List[Tuple[str, ...]]] = {
    "i": [("ɿ", "#")],
}

# 语保 北京-北京-西城
TONE_MAPPING = {
    1: "55",  # ā
    2: "35",  # á
    3: "215",  # ǎ
    4: "51",  # à
    5: "0",  # a
}


def get_tone(pinyin: str) -> int:
    """
    获取拼音的声调编号。
    """
    pinyin_tone3 = to_tone3(pinyin, neutral_tone_with_five=True, v_to_u=True)
    if len(pinyin_tone3) == 0:
        raise ValueError("参数 'pinyin': 无法检测到声调！")

    tone_nr_str = pinyin_tone3[-1]

    try:
        tone_nr = int(tone_nr_str)
    except ValueError as error:
        raise ValueError(f"参数 'pinyin': 声调 '{tone_nr_str}' 无法检测！") from error

    # 防止 to_tone3 返回异常值
    if tone_nr not in TONE_MAPPING:
        raise ValueError(f"参数 'pinyin': 声调 '{tone_nr_str}' 无法检测！")

    return tone_nr


def get_syllabic_consonant(normal_pinyin: str) -> Optional[str]:
    """
    判断是否为音节化辅音。
    """
    if normal_pinyin in SYLLABIC_CONSONANTS:
        return normal_pinyin
    return None


def get_interjection(normal_pinyin: str) -> Optional[str]:
    """
    判断是否为感叹词。
    """
    if normal_pinyin in INTERJECTIONS:
        return normal_pinyin
    return None


def get_initials(normal_pinyin: str) -> Optional[str]:
    """
    获取拼音的声母。
    """
    if normal_pinyin in SYLLABIC_CONSONANTS:
        return None

    if normal_pinyin in INTERJECTIONS:
        return None

    pinyin_initial = to_initials(normal_pinyin, strict=True)

    if pinyin_initial == "":
        return None

    # 防止 pypinyin 返回异常值
    if pinyin_initial not in INITIAL_MAPPING:
        raise ValueError(f"参数 'normal_pinyin': 声母 '{pinyin_initial}' 无法检测！")

    return pinyin_initial


def get_finals(normal_pinyin: str) -> Optional[str]:
    """
    获取拼音的韵母。
    """
    if normal_pinyin in SYLLABIC_CONSONANTS:
        return None

    if normal_pinyin in INTERJECTIONS:
        return None

    pinyin_final = to_finals(normal_pinyin, strict=True, v_to_u=True)

    if pinyin_final == "":
        raise ValueError("参数 'normal_pinyin': 无法检测到韵母！")

    # 防止 pypinyin 返回异常值
    if pinyin_final not in FINAL_MAPPING:
        raise ValueError(f"参数 'normal_pinyin': 韵母 '{pinyin_final}' 无法检测！")

    return pinyin_final


def apply_tone(
    variants: List[Tuple[str, ...]], tone: int
) -> Generator[Tuple[str, ...], None, None]:
    """
    将声调应用到音素变体上。
    """
    tone_ipa = TONE_MAPPING[tone]
    yield from (
        tuple(phoneme.replace("#", tone_ipa) for phoneme in variant) for variant in variants
    )


def pinyin_to_ipa(pinyin: str) -> str:
    """
    将拼音音节转换为对应的国际音标（IPA）转写。

    参数
    ----------
    pinyin : str
        需要转写为IPA的拼音音节。输入可以包含声调标记（如 "zhong", "zhōng", "zho1ng", "zhong1"）。

    返回
    -------
    str
        输入拼音的IPA转写字符串。

    异常
    ------
    ValueError
        如果无法检测到声调，或声母/韵母无法映射到IPA，则抛出异常。

    说明
    -----
    - 支持感叹词和音节化辅音等特殊情况，这些不严格属于声母-韵母结构。
    - 声调会加在元音或音节化辅音上。
    - 依赖 `pypinyin` 库进行声母和韵母的分割。

    示例
    --------
    将带声调的拼音字符串转为IPA：

    >>> result = pinyin_to_ipa("zhong4")
    >>> print(result)
    tʂuŋ51
    """
    tone_nr = get_tone(pinyin)
    pinyin_normal = to_normal(pinyin)

    interjection = get_interjection(pinyin_normal)
    if interjection is not None:
        interjection_ipa_mapping = INTERJECTION_MAPPINGS[pinyin_normal]
        interjection_ipa = OrderedSet(apply_tone(interjection_ipa_mapping, tone_nr))
        combination = interjection_ipa[0]
        return "".join(combination)

    syllabic_consonant = get_syllabic_consonant(pinyin_normal)
    if syllabic_consonant is not None:
        syllabic_consonant_ipa_mapping = SYLLABIC_CONSONANT_MAPPINGS[syllabic_consonant]
        syllabic_consonant_ipa = OrderedSet(apply_tone(syllabic_consonant_ipa_mapping, tone_nr))
        combination = syllabic_consonant_ipa[0]
        return "".join(combination)

    parts = []
    pinyin_initial = get_initials(pinyin_normal)
    pinyin_final = get_finals(pinyin_normal)
    assert pinyin_final is not None

    if pinyin_initial is not None:
        initial_phonemes = INITIAL_MAPPING[pinyin_initial]
        parts.append(initial_phonemes)

    final_phonemes: List[Tuple[str, ...]]
    if pinyin_initial in {"zh", "ch", "sh", "r"} and pinyin_final in FINAL_MAPPING_AFTER_ZH_CH_SH_R:
        final_phonemes = FINAL_MAPPING_AFTER_ZH_CH_SH_R[pinyin_final]
    elif pinyin_initial in {"z", "c", "s"} and pinyin_final in FINAL_MAPPING_AFTER_Z_C_S:
        final_phonemes = FINAL_MAPPING_AFTER_Z_C_S[pinyin_final]
    else:
        final_phonemes = FINAL_MAPPING[pinyin_final]

    final_phonemes = list(apply_tone(final_phonemes, tone_nr))
    parts.append(final_phonemes)

    assert len(parts) >= 1

    all_syllable_combinations = OrderedSet(
        tuple(itertools.chain.from_iterable(combination))
        for combination in itertools.product(*parts)
    )

    combination = all_syllable_combinations[0]

    return "".join(combination)


# if __name__ == "__main__":
#     print(pinyin_to_ipa("wa1"))
#     print(pinyin_to_ipa("ya1"))
#     print(pinyin_to_ipa("bo1"))
#     print(pinyin_to_ipa("si1"))
#     print(pinyin_to_ipa("shi1"))
#     print(pinyin_to_ipa("zhuang1"))
#     print(pinyin_to_ipa("weng1"))
#     print(pinyin_to_ipa("wong1"))
