import argparse
import logging
from goruut_ipa import (
    get_ipa_with_segmentation,
    goruut_server_context,
    GoruutConfig,
)

# Configure logging
import os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/direct_ipa.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def direct_ipa_conversion(
    language: str, sentence: str, config: GoruutConfig = None
) -> bool:
    """Convert a given sentence to IPA, report failed words and success status."""
    if not sentence.strip():
        logger.error("输入句子为空")
        print("错误：输入句子为空")
        return False

    with goruut_server_context(config) as server:
        try:
            ipa, failed_words, success = get_ipa_with_segmentation(
                language, sentence, server.config
            )
            if success:
                print(f"IPA: {ipa}")
                return True
            else:
                print(
                    f"错误：无法为句子 '{sentence}' 获取 IPA，失败词: {failed_words}"
                )
                logger.error(f"无法获取 IPA 音标，失败词: {failed_words}")
                return False
        except Exception as e:
            print(f"错误：转写 IPA 失败: {e}")
            logger.error(f"转写 IPA 失败：{e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="将输入句子转为 IPA")
    parser.add_argument(
        "--language", required=True, help="语言（例如：Chinese Mandarin）"
    )
    parser.add_argument("--sentence", required=True, help="要转写的句子")

    args = parser.parse_args()

    config = GoruutConfig()
    if not direct_ipa_conversion(args.language, args.sentence, config):
        exit(1)


if __name__ == "__main__":
    main()
