import argparse
import logging
from goruut_ipa import run_goruut_and_get_ipa, GoruutConfig

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


def main():
    parser = argparse.ArgumentParser(description="Convert text to IPA")
    parser.add_argument(
        "--language", required=True, help="(eg. Mandarin)"
    )
    parser.add_argument("--sentence", required=True, help="要转写的句子")

    args = parser.parse_args()

    config = GoruutConfig()
    if not direct_ipa_conversion(args.language, args.sentence, config):
        exit(1)


if __name__ == "__main__":
    main()
