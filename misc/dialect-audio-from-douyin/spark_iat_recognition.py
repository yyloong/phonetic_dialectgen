import os
import sys
import time
import json
import base64
import hashlib
import hmac
import ssl
import datetime
import threading
import websocket
from pathlib import Path
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from time import mktime
from collections import defaultdict
import re

# 全局变量
LOG_FILE = None
processed_files = set()
failed_files = []
successful_files = []
processing_results = {}
raw_responses = {}  # 保存原始响应数据

# 处理状态
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2


def print_and_write_log(message):
    """统一的打印和写入日志接口"""
    print(message)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def load_processed_files():
    """加载已处理文件列表"""
    progress_file = Path("spark_iat_progress.json")
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed_files", [])), data.get("results", {})
        except Exception as e:
            print_and_write_log(f"加载进度文件失败: {e}")
    return set(), {}


def load_raw_responses():
    """加载原始响应数据"""
    raw_file = Path("spark_iat_progress_raw.json")
    if raw_file.exists():
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("raw_responses", {})
        except Exception as e:
            print_and_write_log(f"加载原始响应数据失败: {e}")
    return {}


def save_processed_files():
    """保存已处理文件列表"""
    progress_file = Path("spark_iat_progress.json")
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed_files": list(processed_files),
                    "results": processing_results,
                    "last_update": datetime.datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print_and_write_log(f"保存进度文件失败: {e}")


def save_raw_responses():
    """保存原始响应数据"""
    raw_file = Path("spark_iat_progress_raw.json")
    try:
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_responses": raw_responses,
                    "last_update": datetime.datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print_and_write_log(f"保存原始响应数据失败: {e}")


def group_files_by_timestamp(files):
    """根据时间戳对文件进行分组"""
    groups = defaultdict(list)
    for file in files:
        match = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_PART(\d+)\.wav", file.name)
        if match:
            timestamp = match.group(1)
            part_num = int(match.group(2))
            groups[timestamp].append((part_num, file))

    # 对每个组内的文件按PART号排序
    for timestamp in groups:
        groups[timestamp].sort(key=lambda x: x[0])

    return groups


def determine_status(part_num, max_part):
    """根据PART号确定状态"""
    if part_num == 0:
        return STATUS_FIRST_FRAME
    elif part_num == max_part:
        return STATUS_LAST_FRAME
    else:
        return STATUS_CONTINUE_FRAME


class SparkIATClient:
    def __init__(self, app_id, api_key, api_secret):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws = None
        self.result_text = ""
        self.error_message = ""
        self.is_finished = False
        self.lock = threading.Lock()
        
        # 保存详细的识别结果
        self.detailed_results = []
        self.raw_messages = []  # 保存原始响应消息
        self.current_group_key = None

        # 配置参数
        self.iat_params = {
            "language": "zh_cn",
            "accent": "mulacc",
            "domain": "slm",
            "eos": 1800,
            "nbest": 5,
            "wbest": 5,
            "vinfo": 1,
            "ptt": 1,
            "nunum": 0,
            "opt": 2,
            "rlang": "zh-cn",
            "ltc": 1,
            "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
        }

    def create_url(self):
        """生成带鉴权的WebSocket URL"""
        url = "wss://iat.cn-huabei-1.xf-yun.com/v1"
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 按照官方文档的格式构建签名字符串
        signature_origin = "host: " + "iat.cn-huabei-1.xf-yun.com" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v1 " + "HTTP/1.1"

        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )

        params = {
            "authorization": authorization,
            "date": date,
            "host": "iat.cn-huabei-1.xf-yun.com",
        }

        return url + "?" + urlencode(params)

    def on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            message_data = json.loads(message)
            code = message_data["header"]["code"]
            status = message_data["header"]["status"]

            # 保存原始消息
            timestamp = datetime.datetime.now().isoformat()
            raw_message_record = {
                "timestamp": timestamp,
                "raw_message": message_data,
                "group_key": self.current_group_key
            }
            self.raw_messages.append(raw_message_record)

            if code != 0:
                self.error_message = (
                    f"API错误码: {code}, 消息: {message_data['header'].get('message', '')}"
                )
                self.is_finished = True
                return

            payload = message_data.get("payload")
            if payload and payload.get("result"):
                text = payload["result"]["text"]
                if text:
                    decoded_text = json.loads(str(base64.b64decode(text), "utf8"))
                    
                    # 保存完整的解码后文本结构
                    detailed_result = {
                        "timestamp": timestamp,
                        "decoded_text": decoded_text,
                        "bg": decoded_text.get("bg", 0),  # 开始时间
                        "ed": decoded_text.get("ed", 0),  # 结束时间
                        "ls": decoded_text.get("ls", False),  # 是否最后一块
                        "sn": decoded_text.get("sn", 0),  # 序号
                        "pgs": decoded_text.get("pgs", ""),  # 流式识别操作方式
                        "rst": decoded_text.get("rst", ""),  # 识别结果类型
                        "rg": decoded_text.get("rg", []),  # 结果标识
                        "ws": decoded_text.get("ws", []),  # 完整的词信息
                        "group_key": self.current_group_key
                    }
                    self.detailed_results.append(detailed_result)

                    # 提取文本内容（保持原有逻辑）
                    text_ws = decoded_text.get("ws", [])
                    partial_result = ""
                    for i in text_ws:
                        for j in i.get("cw", []):
                            w = j.get("w", "")
                            partial_result += w

                    with self.lock:
                        self.result_text += partial_result

            if status == 2:
                self.is_finished = True

        except Exception as e:
            self.error_message = f"解析消息失败: {e}"
            self.is_finished = True

    def on_error(self, ws, error):
        """处理WebSocket错误"""
        self.error_message = f"WebSocket错误: {error}"
        self.is_finished = True

    def on_close(self, ws, close_status_code, close_msg):
        """处理WebSocket关闭"""
        self.is_finished = True

    def on_open(self, ws):
        """处理WebSocket连接建立"""

        def run(*args):
            try:
                frame_size = 1280
                interval = 0.05

                # 按顺序处理所有音频文件
                for part_num, audio_file in self.audio_files:
                    # 根据PART号确定状态
                    part_status = determine_status(part_num, self.max_part)
                    seq = part_num

                    with open(audio_file, "rb") as fp:
                        is_first_chunk = True
                        while True:
                            buf = fp.read(frame_size)
                            if not buf:
                                break

                            audio_data = str(base64.b64encode(buf), "utf-8")

                            # 构建请求数据
                            if part_status == STATUS_FIRST_FRAME and is_first_chunk:
                                # 第一个PART的第一帧，包含完整参数
                                data = {
                                    "header": {"app_id": self.app_id, "status": part_status},
                                    "parameter": {"iat": self.iat_params},
                                    "payload": {
                                        "audio": {
                                            "encoding": "raw",
                                            "sample_rate": 16000,
                                            "channels": 1,
                                            "bit_depth": 16,
                                            "status": part_status,
                                            "seq": seq,
                                            "audio": audio_data,
                                        }
                                    },
                                }
                                is_first_chunk = False
                            else:
                                # 其他情况
                                header_status = 1 if part_status != STATUS_LAST_FRAME else 1
                                audio_status = part_status

                                data = {
                                    "header": {"app_id": self.app_id, "status": header_status},
                                    "payload": {
                                        "audio": {
                                            "encoding": "raw",
                                            "sample_rate": 16000,
                                            "channels": 1,
                                            "bit_depth": 16,
                                            "status": audio_status,
                                            "seq": seq,
                                            "audio": audio_data,
                                        }
                                    },
                                }

                            ws.send(json.dumps(data))
                            time.sleep(interval)

                # 发送结束帧（空数据）
                final_data = {
                    "header": {"app_id": self.app_id, "status": 2},
                    "payload": {
                        "audio": {
                            "encoding": "raw",
                            "sample_rate": 16000,
                            "channels": 1,
                            "bit_depth": 16,
                            "status": 2,
                            "seq": self.max_part,
                            "audio": "",
                        }
                    },
                }
                ws.send(json.dumps(final_data))

            except Exception as e:
                self.error_message = f"发送音频数据失败: {e}"
                self.is_finished = True

        threading.Thread(target=run).start()

    def recognize_audio_group(self, audio_files, group_key, max_retries=3):
        """识别一组音频文件"""
        self.audio_files = audio_files
        self.max_part = max([part_num for part_num, _ in audio_files])
        self.current_group_key = group_key

        for attempt in range(max_retries):
            try:
                # 重置状态
                self.result_text = ""
                self.error_message = ""
                self.is_finished = False
                self.detailed_results = []
                self.raw_messages = []

                # 创建WebSocket连接
                ws_url = self.create_url()
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                self.ws.on_open = self.on_open

                # 启动连接
                self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

                # 等待处理完成
                while not self.is_finished:
                    time.sleep(0.1)

                if self.error_message:
                    print_and_write_log(
                        f"识别失败 (尝试 {attempt + 1}/{max_retries}): {self.error_message}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # 指数退避
                        continue
                    else:
                        return None, self.error_message, [], []
                else:
                    return self.result_text, None, self.detailed_results, self.raw_messages

            except Exception as e:
                error_msg = f"连接失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                print_and_write_log(error_msg)
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    return None, error_msg, [], []

        return None, "超过最大重试次数", [], []


def process_directory(directory, client: SparkIATClient):
    """处理目录中的音频文件"""
    directory = Path(directory)
    if not directory.exists():
        print_and_write_log(f"目录不存在: {directory}")
        return

    print_and_write_log(f"开始处理目录: {directory}")

    # 获取所有wav文件
    wav_files = list(directory.glob("*.wav"))
    print_and_write_log(f"找到 {len(wav_files)} 个wav文件")

    # 按时间戳分组
    file_groups = group_files_by_timestamp(wav_files)
    print_and_write_log(f"分组后有 {len(file_groups)} 个时间戳组")

    # 处理每个组
    for timestamp, files in file_groups.items():
        group_key = f"{directory.name}_{timestamp}"

        # 检查是否已处理
        if group_key in processed_files:
            print_and_write_log(f"跳过已处理的组: {group_key}")
            continue

        print_and_write_log(f"处理组: {group_key} (包含 {len(files)} 个文件)")

                # 识别音频
        result, error, detailed_results, raw_messages = client.recognize_audio_group(files, group_key)

        if result:
            print_and_write_log(f"识别成功: {group_key}")
            print_and_write_log(f"识别结果: {result}")

            # 保存结果
            processing_results[group_key] = {
                "result": result,
                "files": [f.name for _, f in files],
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "success",
                "detailed_results": detailed_results,  # 保存详细结果
            }

            # 保存原始响应数据
            raw_responses[group_key] = {
                "files": [f.name for _, f in files],
                "timestamp": datetime.datetime.now().isoformat(),
                "detailed_results": detailed_results,
                "raw_messages": raw_messages,
            }

            processed_files.add(group_key)
            successful_files.append(group_key)
            
            # 立即保存进度和原始数据
            save_processed_files()
            save_raw_responses()
        else:
            print_and_write_log(f"识别失败: {group_key} - {error}")

            # 保存失败信息
            processing_results[group_key] = {
                "error": error,
                "files": [f.name for _, f in files],
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "failed",
            }

            # 即使失败也保存原始响应数据（可能包含部分信息）
            raw_responses[group_key] = {
                "files": [f.name for _, f in files],
                "timestamp": datetime.datetime.now().isoformat(),
                "error": error,
                "detailed_results": detailed_results,
                "raw_messages": raw_messages,
            }

            failed_files.append(group_key)
            
            # 立即保存进度（包括失败的记录）
            save_processed_files()
            save_raw_responses()

        # 添加延迟避免API限制
        time.sleep(1)


def main():
    """主函数"""
    global LOG_FILE, processed_files, processing_results, raw_responses

    # 设置日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"spark_iat_recognition_{timestamp}.log"

    print_and_write_log("开始讯飞语音识别任务")

    # 加载已处理文件
    processed_files, processing_results = load_processed_files()
    raw_responses = load_raw_responses()
    print_and_write_log(f"加载了 {len(processed_files)} 个已处理文件记录")
    print_and_write_log(f"加载了 {len(raw_responses)} 个原始响应数据记录")

    # # API配置 - 需要用户填写
    # app_id = "*********"  # 请填写您的APPID
    # api_key = "********************"  # 请填写您的APIKey
    # api_secret = "********************"  # 请填写您的APISecret

    # if (
    #     app_id == "*********"
    #     or api_key == "********************"
    #     or api_secret == "********************"
    # ):
    #     print_and_write_log("错误: 请先配置API密钥信息")
    #     return

    app_id = input("请输入APPID: ").strip()
    api_key = input("请输入APIKey: ").strip()
    api_secret = input("请输入APISecret: ").strip()

    # 创建客户端
    client = SparkIATClient(app_id, api_key, api_secret)

    # 处理目录
    directories = ["shorts", "shorts_noisy"]

    for directory in directories:
        if Path(directory).exists():
            process_directory(directory, client)
        else:
            print_and_write_log(f"目录不存在: {directory}")

    # 输出统计信息
    print_and_write_log(f"处理完成!")
    print_and_write_log(f"成功处理: {len(successful_files)} 个组")
    print_and_write_log(f"失败: {len(failed_files)} 个组")

    if failed_files:
        print_and_write_log("失败的文件组:")
        for failed_file in failed_files:
            print_and_write_log(f"  - {failed_file}")

    # 保存最终结果
    results_file = f"spark_iat_results_{timestamp}.json"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(processing_results, f, ensure_ascii=False, indent=2)
        print_and_write_log(f"结果已保存到: {results_file}")
    except Exception as e:
        print_and_write_log(f"保存结果文件失败: {e}")
    
    # 保存原始响应数据的最终副本
    raw_results_file = f"spark_iat_raw_results_{timestamp}.json"
    try:
        with open(raw_results_file, "w", encoding="utf-8") as f:
            json.dump(raw_responses, f, ensure_ascii=False, indent=2)
        print_and_write_log(f"原始响应数据已保存到: {raw_results_file}")
    except Exception as e:
        print_and_write_log(f"保存原始响应数据失败: {e}")
    
    # 数据完整性总结
    print_and_write_log("=" * 60)
    print_and_write_log("数据保存总结:")
    print_and_write_log(f"基础结果文件: {results_file}")
    print_and_write_log(f"原始响应数据文件: {raw_results_file}")
    print_and_write_log(f"进度跟踪文件: spark_iat_progress.json")
    print_and_write_log(f"原始数据跟踪文件: spark_iat_progress_raw.json")
    print_and_write_log("=" * 60)
    print_and_write_log("保存的详细信息包括:")
    print_and_write_log("✓ 每句话的开始和结束时间戳 (bg, ed)")
    print_and_write_log("✓ 句子和词的多候选结果 (ws结构)")
    print_and_write_log("✓ 标点符号和词性信息")
    print_and_write_log("✓ 流式识别的完整状态信息")
    print_and_write_log("✓ 完整的原始HTTP响应数据")
    print_and_write_log("这些数据可用于后续的语音AI模型训练!")
    print_and_write_log("=" * 60)


if __name__ == "__main__":
    main()
