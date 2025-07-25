import logging
import re
from typing import Tuple, List
import ToJyutping
from pypinyin import lazy_pinyin, Style
from .language.jyutping import jyutping_to_ipa
from .language.pinyin import pinyin_to_ipa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Jyutping validation regex
jyutping_regex = re.compile(
    r"^([gk]w?|ng|[bpmfdtnlhwzcsj]?)(?![1-6]?$)((aa?|oe?|eo?|y?u|i?)(ng|[iumnptk]?))([1-6]?)$"
)


def text_to_IPA(text: str, language: str) -> Tuple[str, List[str], bool]:
    """Convert Chinese text to IPA based on specified language (pinyin or jyutping), retaining punctuation.

    Args:
        text (str): Input Chinese text.
        language (str): Either 'pinyin' for Mandarin or 'jyutping' for Cantonese.

    Returns:
        Tuple[str, List[str], bool]: IPA string, list of failed characters, and success flag.
    """
    if not text.strip():
        logger.error("输入文本为空")
        return "", [], False

    if language not in ["pinyin", "jyutping"]:
        logger.error(
            f"不支持的语言类型: {language}，必须是 'pinyin' 或 'jyutping'"
        )
        raise ValueError("Language must be 'pinyin' or 'jyutping'")

    # Filter out non-Chinese characters
    hanzi_pattern = re.compile(r"[\u4e00-\u9fff]")

    ipa_parts = []
    failed_words = []
    success = True

    for char in text:
        if not hanzi_pattern.match(char):
            ipa_parts.append(char)  # Retain punctuation
            logger.info(f"保留标点: {char}")
            continue

        try:
            if language == "jyutping":
                # Get Jyutping for the character
                jyutping_text = ToJyutping.get_jyutping_text(char)
                jyutping_list = jyutping_text.split() if jyutping_text else []
                if jyutping_list and jyutping_list[0]:
                    jyutping = jyutping_list[0]  # Take the first Jyutping
                    if not jyutping_regex.match(jyutping):
                        logger.warning(
                            f"无效的 Jyutping 格式 '{jyutping}' 对于字符 '{char}'"
                        )
                        failed_words.append(char)
                        success = False
                        ipa_parts.append(char)  # Fallback to character
                        continue
                    try:
                        ipa = jyutping_to_ipa(jyutping)
                        ipa_parts.append(ipa)
                        logger.info(
                            f"成功获取 IPA: {char} -> {jyutping} -> {ipa}"
                        )
                    except ValueError as e:
                        logger.warning(
                            f"无法转换 Jyutping '{jyutping}' 到 IPA: {e}"
                        )
                        failed_words.append(char)
                        success = False
                        ipa_parts.append(jyutping)  # Fallback to Jyutping
                else:
                    logger.warning(
                        f"无法为字符 '{char}' 获取粤语拼音，可能为非粤语字符"
                    )
                    failed_words.append(char)
                    success = False
                    ipa_parts.append(char)  # Fallback to character
            else:  # language == "pinyin"
                # Get pinyin for the character
                pinyin_list = lazy_pinyin(
                    char, style=Style.TONE3, neutral_tone_with_five=True
                )
                if pinyin_list and pinyin_list[0]:
                    pinyin = pinyin_list[0]
                    try:
                        ipa = pinyin_to_ipa(pinyin)
                        ipa_parts.append(ipa)
                        logger.info(
                            f"成功获取 IPA: {char} -> {pinyin} -> {ipa}"
                        )
                    except ValueError as e:
                        logger.warning(f"无法转换拼音 '{pinyin}' 到 IPA: {e}")
                        failed_words.append(char)
                        success = False
                        ipa_parts.append(pinyin)  # Fallback to pinyin
                else:
                    logger.warning(f"无法为字符 '{char}' 获取拼音")
                    failed_words.append(char)
                    success = False
                    ipa_parts.append(char)  # Fallback to character
        except Exception as e:
            logger.error(f"处理字符 '{char}' 失败: {e}")
            failed_words.append(char)
            success = False
            ipa_parts.append(char)  # Fallback to character

    # Join IPA parts with spaces, retaining punctuation
    result = " ".join(ipa_parts).strip()
    success = len(failed_words) == 0 and result.strip() != ""
    logger.info(
        f"最终 IPA: {text} -> {result}, 失败字符: {failed_words}, 成功: {success}"
    )
    return result, failed_words, success