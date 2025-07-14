import requests
import subprocess
import socket
import time
import os
import logging
import json
import jieba
import re
from contextlib import contextmanager
from pypinyin import pinyin, Style
from typing import Tuple, List

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/goruut_ipa.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load tone-to-number mapping from unicodes_tone_to_numbers.txt
TONE_MAP = {}
try:
    with open("unicodes_tone_to_numbers.txt", "r", encoding="utf-8") as f:
        for line in f:
            tone, number = line.strip().split()
            TONE_MAP[tone] = number
except FileNotFoundError:
    logger.error("找不到 unicodes_tone_to_numbers.txt")
    raise FileNotFoundError("unicodes_tone_to_numbers.txt not found")

# Create regex pattern for tones (covering all tones from unicodes_tone_to_numbers.txt)
# Ensure multi-character tones (e.g., ˥˥, ˥˩) are matched as single units
tone_pattern = re.compile(
    r"(?:(?:[\u02E5-\u02E9]{1,3})(?=\s|$|[^\u02E5-\u02E9]))", re.UNICODE
)

# Regex pattern for punctuation (remove only Chinese and English punctuation, preserve IPA and tones)
# Allow: a-z, IPA characters (ɐ-ʸ), tone symbols (˥, ˩, etc.), spaces
# Remove: Chinese punctuation (\u3000-\u303F), English punctuation (,.!?;:"'[](){})
punctuation_pattern = re.compile(r"[^\w\sɐ-ʸ\u02E5-\u02E9]")

# Common homophone pool for fallback
HOMOPHONE_POOL = {
    "yǒu": ["友", "由", "右"],
    "wài": ["外", "歪"],
    "zhì": ["置", "制", "智"],
    "zài": ["再", "载"],
    "shì": ["试", "事"],
    "de": ["地", "德"],
    "dé": ["得", "德"],
    "shè": ["设", "社"],
    "bèi": ["备", "背"],
    "xiàng": ["像", "相"],
    "tóu": ["头", "投"],
    "shí": ["识", "实"],
    "zhuì": ["坠", "赘"],
    "zhe": ["者", "折"],
    "zháo": ["找", "灼"],
    "zhāo": ["招", "朝"],
    "dì": ["弟", "帝"],
    "dí": ["敌", "迪"],
    "zhī": ["支", "之"],
    "wú": ["吴", "巫"],
    "bá": ["霸", "把"],
    "dǎ": ["大", "搭"],
    "cǎo": ["操", "曹"],
    "hé": ["合", "河"],
    "huò": ["或", "获"],
    "huó": ["活", "火"],
    "huo": ["和", "或"],
    "hóng": ["宏", "洪"],
    "le": ["乐", "勒"],
    "liǎo": ["料", "瞭"],
    "mò": ["莫", "末"],
    "sè": ["瑟", "塞"],
    "shǎi": ["晒", "筛"],
    "měi": ["美", "妹"],
    "hǎo": ["号", "浩"],
    "wàn": ["腕", "万"],
    "chuán": ["船", "川"],
    "qīng": ["青", "轻"],
    "bǐ": ["笔", "彼"],
    "gèng": ["耿", "更"],
    "diǎn": ["典", "电"],
    "xiān": ["仙", "先"],
    "yàn": ["燕", "厌"],
    "cǎi": ["采", "才"],
    "chuāng": ["床", "创"],
    "cuì": ["翠", "脆"],
    "luò": ["洛", "络"],
    "dōu": ["兜", "都"],
    "lán": ["兰", "岚"],
    "bái": ["百", "柏"],
    "yún": ["云", "匀"],
    "tiān": ["填", "天"],
    "zhuǎn": ["赚", "专"],
    "yǎn": ["演", "掩"],
    "jiān": ["坚", "监"],
    "shùn": ["顺", "舜"],
    "qǐng": ["情", "清"],
    "dá": ["达", "答"],
    "àn": ["按", "安"],
    "dàn": ["丹", "单"],
    "bù": ["布", "部"],
    "shèng": ["盛", "胜"],
    "shōu": ["搜", "收"],
    "màn": ["慢", "曼"],
    "yè": ["夜", "叶"],
    "piāo": ["漂", "票"],
    "chōng": ["冲", "充"],
    "jǐng": ["景", "净"],
    "chuī": ["炊", "吹"],
    "sī": ["丝", "司"],
    "qiān": ["迁", "千"],
    "xù": ["续", "絮"],
    "piàn": ["篇", "片"],
    "shù": ["树", "数"],
    "líng": ["铃", "灵"],
    "shēn": ["身", "深"],
    "chù": ["触", "处"],
    "yǎng": ["羊", "扬"],
    "chǎng": ["厂", "场"],
    "yáng": ["扬", "羊"],
    "bàn": ["半", "伴"],
    "jiā": ["家", "佳", "嘉"],
    "liàng": ["亮", "量"],
    "jiē": ["街", "接"],
    "kāi": ["开", "凯"],
    "huā": ["花", "华"],
    "bié": ["别", "憋"],
    "cì": ["刺", "次"],
    "bān": ["班", "斑"],
    "zǒng": ["总", "宗"],
    "néng": ["能", "农"],
    "chí": ["池", "持"],
    "xiè": ["谢", "泄"],
    "rì": ["日", "入"],
    "biān": ["边", "鞭"],
    "qīn": ["亲", "勤"],
    "rén": ["人", "仁"],
    "zhè": ["浙", "遮"],
    "yī": ["衣", "依"],
    "fèn": ["分", "奋"],
    "bēn": ["贲", "奔"],
    "pǎo": ["泡", "袍"],
    "zhōng": ["忠", "钟"],
    "fú": ["福", "伏"],
    "gè": ["各", "戈"],
    "cháng": ["常", "场"],
    "yè": ["液", "叶"],
    "màn": ["曼", "慢"],
    "zhēng": ["征", "睁"],
    "zú": ["族", "卒"],
    "cuǐ": ["翠", "萃"],
    "càn": ["灿", "粲"],
    "mái": ["霾", "霉"],
    "cuò": ["措", "挫"],
    "kā": ["卡", "喀"],
    "fēi": ["飞", "菲"],
    "kè": ["客", "刻"],
    "nán": ["南", "男"],
    "fú": ["福", "伏"],
    "kùn": ["昆", "坤"],
    "zhōng": ["钟", "忠"],
    "jiāng": ["江", "疆"],
    "huì": ["汇", "绘"],
    "xī": ["溪", "熙"],
    "lì": ["历", "砾"],
    "wǎn": ["宛", "晚"],
    "zhuàn": ["赚", "撰"],
    "bà": ["霸", "吧"],
    "shì": ["势", "试"],
    "xiū": ["修", "羞"],
    "shēn": ["申", "伸"],
    "cáng": ["仓", "舱"],
    "zhèng": ["证", "症"],
    "zài": ["再", "载"],
    "shuō": ["朔", "硕"],
    "sù": ["速", "塑"],
    "yì": ["亿", "毅"],
    "huī": ["灰", "徽"],
    "shēng": ["声", "升"],
    "miǎo": ["藐", "秒"],
    "piāo": ["漂", "票"],
    "gài": ["概", "丐"],
    "hú": ["胡", "湖"],
    "dié": ["叠", "碟"],
    "zhǒng": ["肿", "冢"],
    "huǒ": ["伙", "货"],
    "mǎn": ["曼", "漫"],
    "chōng": ["冲", "充"],
    "qíng": ["清", "晴"],
    "rè": ["热", "惹"],
    "sǎ": ["撒", "卅"],
    "áo": ["熬", "嗷"],
    "xiáng": ["祥", "详"],
    "jì": ["纪", "继"],
    "yǒu": ["友", "由"],
    "chūn": ["纯", "醇"],
    "rì": ["入", "日"],
    "zhēn": ["真", "贞"],
    "xìn": ["衅", "鑫"],
    "xīn": ["新", "欣"],
    "yuǎn": ["园", "圆"],
    "lǚ": ["吕", "侣"],
    "pí": ["皮", "疲"],
    "qìn": ["浸", "渗"],
    "rén": ["仁", "刃"],
    "ní": ["妮", "尼"],
    "hóng": ["红", "宏"],
    "guàn": ["罐", "贯"],
    "zhǎ": ["铡", "眨"],
    "zhù": ["注", "驻"],
    "chī": ["痴", "螭"],
    "gāo": ["糕", "膏"],
    "zhòng": ["众", "仲"],
    "wù": ["雾", "务"],
    "xī": ["溪", "熙"],
    "lì": ["历", "砾"],
    "miǎo": ["藐", "秒"],
    "áo": ["熬", "嗷"],
    "xiáng": ["祥", "详"],
    "shǎo": ["哨", "绍"],
    "duō": ["朵", "夺"],
    "chéng": ["城", "承"],
    "dù": ["渡", "杜"],
    "guò": ["郭", "廓"],
    "mī": ["迷", "靡"],
    "zhe": ["者", "折"],
    "jiàng": ["匠", "酱"],
    "lín": ["林", "邻"],
    "mù": ["墓", "暮"],
    "tuì": ["退", "蜕"],
    "sè": ["涩", "塞"],
    "jīng": ["京", "经"],
    "tàn": ["探", "炭"],
    "rén": ["仁", "刃"],
    "tǎng": ["趟", "烫"],
    "dùn": ["顿", "盾"],
    "ér": ["而", "耳"],
    "ne": ["泥", "呢"],
    "hào": ["号", "昊"],
    "hàn": ["汉", "翰"],
    "yàng": ["养", "氧"],
    "zhǒng": ["肿", "冢"],
    "gè": ["阁", "格"],
    "jīng": ["京", "精"],
    "yì": ["义", "艺"],
    "fǎng": ["访", "放"],
    "shēng": ["声", "升"],
    "bì": ["壁", "辟"],
    "pái": ["牌", "排"],
    "huái": ["怀", "淮"],
    "bì": ["必", "毕"],
    "hù": ["户", "湖"],
    "suǒ": ["锁", "琐"],
    "jù": ["具", "剧"],
    "liú": ["刘", "留"],
    "cāo": ["曹", "槽"],
    "kòng": ["空", "孔"],
    "qí": ["其", "奇"],
    "fēi": ["妃", "菲"],
    "qiáng": ["墙", "腔"],
    "huà": ["花", "画"],
    "gé": ["格", "革"],
    "gǔ": ["谷", "鼓"],
    "gǒu": ["狗", "苟"],
    "chuǎn": ["喘", "舛"],
    "yán": ["严", "岩"],
    "cán": ["参", "餐"],
    "wǎng": ["望", "忘"],
    "kuǐ": ["魁", "葵"],
    "lěi": ["蕾", "垒"],
    "chú": ["储", "楚"],
    "gé": ["革", "隔"],
    "bǐ": ["彼", "鄙"],
    "xiè": ["卸", "谢"],
    "gǎng": ["港", "冈"],
    "shǒu": ["首", "受"],
    "wèi": ["卫", "味"],
    "páo": ["袍", "跑"],
    "xiào": ["笑", "效"],
    "dǐ": ["抵", "低"],
    "jí": ["极", "集"],
    "lā": ["拉", "辣"],
    "jī": ["机", "基"],
    "duī": ["堆", "敦"],
    "dǎo": ["道", "岛"],
    "tǒng": ["桶", "通"],
    "zhì": ["智", "志"],
    "xì": ["戏", "细"],
    "fèi": ["废", "沸"],
    "yì": ["意", "艺"],
    "zhù": ["注", "驻"],
    "bù": ["布", "步"],
    "gòu": ["购", "够"],
    "jiàn": ["健", "舰"],
    "chuàn": ["川", "船"],
    "pí": ["脾", "疲"],
    "zhě": ["折", "遮"],
    "shuāi": ["摔", "衰"],
    "tuì": ["褪", "蜕"],
    "tì": ["剃", "替"],
    "chí": ["池", "迟"],
    "jǐng": ["景", "净"],
    "qí": ["齐", "奇"],
    "qū": ["区", "曲"],
    "guǒ": ["裹", "锅"],
    "qí": ["齐", "奇"],
    "jiě": ["姐", "借"],
    "cháng": ["常", "场"],
    "jiǔ": ["酒", "九"],
    "hài": ["海", "骇"],
    "xìng": ["姓", "兴"],
    "xiàn": ["线", "现"],
    "nǔ": ["奴", "怒"],
    "lì": ["立", "历"],
    "shī": ["诗", "狮"],
    "chěng": ["呈", "程"],
    "huó": ["火", "货"],
    "tài": ["太", "泰"],
    "zhì": ["智", "志"],
    "běn": ["奔", "苯"],
    "diǎn": ["典", "店"],
    "shù": ["术", "树"],
    "shēng": ["生", "声"],
    "pǐn": ["聘", "频"],
    "qǐ": ["启", "岂"],
    "wéi": ["维", "围"],
    "zǐ": ["仔", "紫"],
    "jué": ["决", "绝"],
    "zé": ["泽", "责"],
    "fǔ": ["府", "辅"],
    "zhōu": ["州", "洲"],
    "chén": ["臣", "晨"],
    "bó": ["薄", "博"],
    "yán": ["严", "岩"],
    "liào": ["料", "辽"],
    "kào": ["考", "烤"],
    "wèi": ["卫", "味"],
    "míng": ["名", "鸣"],
    "jìn": ["尽", "晋"],
    "huài": ["怀", "淮"],
    "jìn": ["尽", "晋"],
    "huī": ["辉", "徽"],
    "dòng": ["动", "东"],
    "xuán": ["旋", "玄"],
    "wō": ["窝", "蜗"],
    "tù": ["吐", "突"],
    "mī": ["迷", "靡"],
    "jī": ["机", "基"],
    "zhā": ["扎", "渣"],
    "pú": ["浦", "葡"],
    "yīng": ["莹", "莺"],
    "gōng": ["攻", "功"],
    "páng": ["旁", "庞"],
    "xiè": ["卸", "谢"],
    "táo": ["陶", "涛"],
    "xiā": ["霞", "瞎"],
    "wāng": ["亡", "王"],
    "jiào": ["教", "角"],
    "yā": ["压", "亚"],
    "wēng": ["翁", "嗡"],
    "tiǎn": ["添", "甜"],
    "kē": ["科", "柯"],
    "jiǎo": ["角", "脚"],
    "huá": ["滑", "化"],
    "xiǎn": ["显", "县"],
    "jiā": ["加", "佳"],
    "qì": ["汽", "弃"],
    "bāo": ["抱", "饱"],
    "gū": ["孤", "姑"],
    "lū": ["卢", "盧"],
    "lóng": ["龙", "笼"],
    "ěr": ["耳", "而"],
    "lún": ["轮", "仑"],
    "dí": ["笛", "狄"],
    "sī": ["丝", "私"],
    "dùn": ["顿", "盾"],
    "yǒng": ["涌", "永"],
    "dòng": ["洞", "东"],
    "tuī": ["堆", "敦"],
    "mǔ": ["母", "姆"],
    "hēng": ["哼", "衡"],
    "lái": ["来", "赖"],
    "mò": ["墨", "默"],
    "kè": ["客", "刻"],
    "wěn": ["文", "纹"],
    "guān": ["官", "观"],
    "sǒng": ["耸", "宋"],
    "rán": ["燃", "冉"],
    "shòu": ["寿", "授"],
    "gē": ["歌", "割"],
    "lì": ["力", "立"],
    "yào": ["耀", "要"],
    "jì": ["计", "济"],
    "xuàn": ["炫", "玄"],
    "mù": ["墓", "暮"],
    "lián": ["连", "莲"],
    "yī": ["依", "伊"],
    "liè": ["烈", "列"],
    "niè": ["捏", "聂"],
    "pán": ["盘", "潘"],
    "liú": ["刘", "留"],
    "huáng": ["黄", "皇"],
    "yùn": ["韵", "孕"],
    "hán": ["寒", "韩"],
    "pì": ["辟", "避"],
    "nì": ["腻", "昵"],
    "niàng": ["娘", "酿"],
    "zào": ["灶", "躁"],
    "jiāo": ["交", "娇"],
    "tǔ": ["吐", "突"],
    "piàn": ["偏", "篇"],
    "gǔ": ["鼓", "谷"],
    "huò": ["霍", "豁"],
    "qí": ["齐", "奇"],
    "dǎo": ["导", "道"],
    "sǒng": ["悚", "宋"],
    "yún": ["运", "云"],
    "rù": ["辱", "乳"],
    "qiáng": ["墙", "腔"],
    "wēi": ["微", "威"],
    "hǔ": ["虎", "户"],
    "pò": ["魄", "泊"],
    "shèng": ["盛", "胜"],
    "guāng": ["广", "逛"],
    "jí": ["级", "集"],
    "qí": ["齐", "其"],
    "lí": ["离", "梨"],
    "jìn": ["近", "晋"],
    "fǎng": ["仿", "访"],
    "huáng": ["黄", "皇"],
    "ǎi": ["蔼", "霭"],
    "chóu": ["愁", "稠"],
    "chàng": ["畅", "唱"],
    "kāi": ["凯", "楷"],
    "yǐng": ["映", "颖"],
    "pào": ["炮", "跑"],
    "pán": ["盘", "潘"],
    "shān": ["珊", "删"],
    "lǚ": ["侣", "吕"],
    "xiē": ["协", "邪"],
    "máng": ["忙", "芒"],
    "hǎi": ["害", "骇"],
    "dū": ["都", "督"],
    "xiào": ["笑", "哮"],
    "cáng": ["仓", "舱"],
    "dān": ["单", "担"],
    "miào": ["庙", "秒"],
    "wèi": ["卫", "渭"],
    "xǔ": ["许", "绪"],
    "rú": ["儒", "如"],
    "biàn": ["辩", "遍"],
}


class GoruutConfig:
    """Configuration for goruut server."""

    def __init__(self):
        self.goruut_path = os.getenv("GORUUT_PATH", "./go/bin/goruut")
        self.config_path = os.getenv("GORUUT_CONFIG", "configs/config.json")
        self.host = os.getenv("GORUUT_HOST", "127.0.0.1")
        self.port = int(os.getenv("GORUUT_PORT", 18080))
        self.timeout = int(os.getenv("GORUUT_TIMEOUT", 30))
        self.retry_interval = float(os.getenv("GORUUT_RETRY_INTERVAL", 1.0))


class GoruutServer:
    """Manage goruut server lifecycle."""

    def __init__(self, config: GoruutConfig):
        self.config = config
        self.process = None

    def is_port_open(self):
        """Check if the server port is open."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect((self.config.host, self.config.port))
                return True
            except (socket.timeout, ConnectionRefusedError):
                return False

    def start(self):
        """Start the goruut server."""
        if not os.path.isfile(self.config.goruut_path):
            logger.error(f"goruut 可执行文件未找到：{self.config.goruut_path}")
            raise FileNotFoundError(
                f"goruut executable not found: {self.config.goruut_path}"
            )
        if not os.path.isfile(self.config.config_path):
            logger.error(f"配置文件未找到：{self.config.config_path}")
            raise FileNotFoundError(
                f"Config file not found: {self.config.config_path}"
            )

        try:
            self.process = subprocess.Popen(
                [
                    self.config.goruut_path,
                    "-configfile",
                    self.config.config_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.info("正在启动 goruut 服务器...")
            return True
        except Exception as e:
            logger.error(f"启动 goruut 服务器失败：{e}")
            raise RuntimeError(f"Failed to start goruut server: {e}")

    def wait_for_ready(self):
        """Wait for the server to be ready."""
        start_time = time.time()
        while time.time() - start_time < self.config.timeout:
            if self.is_port_open():
                logger.info("goruut 服务器已就绪！")
                return True
            time.sleep(self.config.retry_interval)
        logger.error(f"goruut 服务器未在 {self.config.timeout} 秒内启动")
        raise TimeoutError(
            f"goruut server did not start within {self.config.timeout} seconds"
        )

    def stop(self):
        """Stop the goruut server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                logger.info("goruut 服务器已关闭。")
            except subprocess.TimeoutExpired:
                logger.warning("goruut 服务器未能正常关闭，强制终止。")
                self.process.kill()


@contextmanager
def goruut_server_context(config: GoruutConfig = None):
    """Context manager for goruut server lifecycle."""
    config = config or GoruutConfig()
    server = GoruutServer(config)
    try:
        server.start()
        server.wait_for_ready()
        yield server
    finally:
        server.stop()


def get_ipa_phonemes(
    language: str, word: str, config: GoruutConfig = None
) -> Tuple[str, bool]:
    """Send a request to goruut server to get IPA phonemes for a word."""
    config = config or GoruutConfig()
    url = f"http://{config.host}:{config.port}/tts/phonemize/sentence"
    headers = {"Content-Type": "application/json"}
    data = {"Language": language, "Sentence": word.strip()}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        result = response.json()
        logger.info(f"服务器响应: {result}")

        phonemes = []
        success = True

        for w in result["Words"]:
            phonetic = w.get("Phonetic", "")
            pos_tags = w.get("PosTags")
            clean_word = w.get("CleanWord", "")

            # Failure condition 1: Phonetic is empty
            if not phonetic:
                logger.warning(f"音标为空: {clean_word}")
                success = False
                continue

            # Count Chinese characters in CleanWord
            hanzi_count = len(re.findall(r"[\u4e00-\u9fff]", clean_word))
            # Count tones in Phonetic
            tone_matches = tone_pattern.findall(phonetic)
            tone_count = len(tone_matches)

            # Failure condition 2: PosTags is None and tone count mismatches character count
            if pos_tags is None and hanzi_count != tone_count:
                logger.warning(
                    f"PosTags 为 None 且注音符号数 ({tone_count}) 与汉字数 ({hanzi_count}) 不一致: {clean_word}"
                )
                success = False
                continue

            # If multiple characters and tones match, split IPA by tones
            if hanzi_count > 1 and tone_count == hanzi_count:
                # Split phonetic by tones, preserving tone symbols
                ipa_parts = []
                current_phoneme = ""
                tone_positions = [
                    (m.start(), m.end())
                    for m in tone_pattern.finditer(phonetic)
                ]
                for i, (start, end) in enumerate(tone_positions):
                    # Extract phoneme up to the current tone
                    if i == 0:
                        current_phoneme = phonetic[:end]
                    else:
                        current_phoneme = phonetic[
                            tone_positions[i - 1][1] : end
                        ]
                    ipa_parts.append(current_phoneme)
                    current_phoneme = ""
                # Handle any remaining phoneme after the last tone
                if tone_positions and tone_positions[-1][1] < len(phonetic):
                    ipa_parts.append(phonetic[tone_positions[-1][1] :])
                # Only use split IPA if it matches hanzi_count
                if len(ipa_parts) == hanzi_count:
                    phonemes.extend(ipa_parts)
                    logger.info(f"成功分割 IPA: {clean_word} -> {ipa_parts}")
                else:
                    phonemes.append(phonetic)
                    logger.warning(
                        f"IPA 分割失败，保留原始: {clean_word} -> {phonetic}"
                    )
            else:
                phonemes.append(phonetic)

        ipa = " ".join(phonemes).strip()
        if not ipa and result["Words"]:
            success = False
            logger.warning(f"无有效音标: {word}")

        if success:
            logger.info(f"成功获取 IPA: {language} - {word} -> {ipa}")
        return ipa, success

    except requests.exceptions.ConnectionError:
        logger.error("无法连接到 goruut 服务器")
        raise ConnectionError("Failed to connect to goruut server")
    except requests.exceptions.RequestException as e:
        logger.error(f"请求 goruut 服务器时出错：{e}")
        raise RuntimeError(f"Request to goruut server failed: {e}")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析响应时出错：{e}, 原始响应: {response.text}")
        return "", False


def get_ipa_with_segmentation(
    language: str, sentence: str, config: GoruutConfig = None
) -> Tuple[str, List[str], bool]:
    """Segment Chinese sentence, convert to IPA for Chinese characters, retain original punctuation, convert tones to numbers."""
    if not sentence.strip():
        logger.error("输入句子为空")
        return "", [], False

    # Filter out non-Chinese characters
    hanzi_pattern = re.compile(r"[\u4e00-\u9fff]")
    tokens = list(jieba.tokenize(sentence))
    tokens = sorted(
        tokens, key=lambda x: x[2] - x[1], reverse=True
    )  # Sort by length
    logger.info(f"所有分词组合: {[(t[0], t[1], t[2]) for t in tokens]}")

    words = list(jieba.cut(sentence, cut_all=False))
    logger.info(f"默认分词结果: {words}")

    ipa_parts = []
    failed_words = []
    i = 0
    sentence_len = len(sentence)
    success = True

    while i < sentence_len:
        matched = False
        char = sentence[i]

        # Handle non-Chinese characters (e.g., punctuation)
        if not hanzi_pattern.match(char):
            ipa_parts.append(
                char
            )  # Retain punctuation or non-Chinese characters
            i += 1
            continue

        # Try all possible segments starting at position i
        for token, start, end in tokens:
            if start != i:
                continue
            word = token
            word_len = end - start

            # Only process Chinese words
            if not all(hanzi_pattern.match(c) for c in word):
                continue

            # Try original word
            try:
                ipa, word_success = get_ipa_phonemes(language, word, config)
                if ipa and word_success:
                    # Split IPA into per-character phonemes
                    ipa_list = ipa.split()
                    if len(ipa_list) == len(
                        word
                    ):  # Ensure one IPA per character
                        ipa_parts.extend(
                            ipa_list
                        )  # Add each character's IPA separately
                    else:
                        # Fallback to single character processing if IPA count doesn't match
                        logger.warning(
                            f"词 '{word}' IPA 数量 ({len(ipa_list)}) 与汉字数 ({len(word)}) 不匹配，拆分为单字"
                        )
                        char_ipa_parts = []
                        for char in word:
                            char_ipa, char_success = get_ipa_phonemes(
                                language, char, config
                            )
                            if char_ipa and char_success:
                                char_ipa_parts.append(char_ipa)
                            else:
                                logger.warning(f"无法为单字 '{char}' 获取 IPA")
                                char_ipa_parts.append(
                                    ""
                                )  # Placeholder for failed character
                        ipa_parts.extend([p for p in char_ipa_parts if p])
                    i += word_len
                    matched = True
                    break
                logger.warning(f"无法为词 '{word}' 获取 IPA")
            except Exception as e:
                logger.error(f"处理词 '{word}' 失败: {e}")

            # Split into characters if word has multiple characters
            if len(word) > 1:
                logger.info(f"拆分词 '{word}' 为单字")
                char_ipa_parts = []
                char_failed = False
                for char in word:
                    try:
                        ipa, char_success = get_ipa_phonemes(
                            language, char, config
                        )
                        if ipa and char_success:
                            char_ipa_parts.append(ipa)
                            continue
                        logger.warning(f"无法为单字 '{char}' 获取 IPA")
                    except Exception as e:
                        logger.error(f"处理单字 '{char}' 失败: {e}")

                    # Try homophone
                    pinyin_tone = "".join(pinyin(char, style=Style.TONE)[0])
                    logger.info(f"单字 '{char}' 拼音: {pinyin_tone}")
                    substitutes = HOMOPHONE_POOL.get(pinyin_tone, [])
                    for substitute in substitutes:
                        try:
                            ipa, sub_success = get_ipa_phonemes(
                                language, substitute, config
                            )
                            if ipa and sub_success:
                                char_ipa_parts.append(ipa)
                                logger.info(
                                    f"成功使用同音字 '{substitute}' 获取 IPA: {ipa}"
                                )
                                break
                        except Exception as e:
                            logger.error(f"处理同音字 '{substitute}' 失败: {e}")
                    else:
                        logger.warning(f"单字 '{char}' 无可用同音字，忽略")
                        char_failed = True
                        failed_words.append(char)

                if char_ipa_parts and not char_failed:
                    ipa_parts.extend(char_ipa_parts)
                    i += word_len
                    matched = True
                    break
                else:
                    logger.warning(f"词 '{word}' 单字拆分失败，记录为失败词")
                    failed_words.append(word)

            # Try homophone for single character
            if len(word) == 1:
                pinyin_tone = "".join(pinyin(word, style=Style.TONE)[0])
                logger.info(f"词 '{word}' 拼音: {pinyin_tone}")
                substitutes = HOMOPHONE_POOL.get(pinyin_tone, [])
                for substitute in substitutes:
                    try:
                        ipa, sub_success = get_ipa_phonemes(
                            language, substitute, config
                        )
                        if ipa and sub_success:
                            ipa_parts.append(ipa)
                            i += word_len
                            matched = True
                            logger.info(
                                f"成功使用同音字 '{substitute}' 获取 IPA: {ipa}"
                            )
                            break
                    except Exception as e:
                        logger.error(f"处理同音字 '{substitute}' 失败: {e}")
                else:
                    logger.warning(f"词 '{word}' 无可用同音字，忽略")
                    failed_words.append(word)

        if not matched:
            logger.warning(f"无法匹配词 at position {i}: '{char}'，忽略")
            failed_words.append(char)
            i += 1

    # Join IPA parts with spaces, retaining punctuation
    result = " ".join(ipa_parts).strip()
    # Convert all tone symbols to numbers in the final output
    # result = re.sub(tone_pattern, lambda m: TONE_MAP.get(m.group(0), m.group(0)), result)
    # result = result.replace("˥˥", "55")
    success = len(failed_words) == 0 and result.strip() != ""
    logger.info(
        f"最终 IPA: {sentence} -> {result}, 失败词: {failed_words}, 成功: {success}"
    )
    return result, failed_words, success


def run_goruut_and_get_ipa(
    language: str, sentence: str, config: GoruutConfig = None
) -> Tuple[str, List[str], bool]:
    """Run goruut server and get IPA phonemes with segmentation."""
    with goruut_server_context(config) as server:
        return get_ipa_with_segmentation(language, sentence, server.config)
