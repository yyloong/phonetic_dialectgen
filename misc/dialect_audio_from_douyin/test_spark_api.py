"""
讯飞API连接测试脚本
用于验证API配置是否正确
"""

import json
import base64
import hashlib
import hmac
import ssl
import datetime
import threading
import websocket
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from time import mktime
import time


class SparkAPITest:
    def __init__(self, app_id, api_key, api_secret):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_result = None
        self.error_message = None
        self.is_finished = False

    def create_url(self):
        """生成带鉴权的WebSocket URL"""
        url = "wss://iat.cn-huabei-1.xf-yun.com/v1"
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        # 按照官方文档的格式构建签名字符串
        signature_origin = "host: " + "iat.cn-huabei-1.xf-yun.com" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v1 " + "HTTP/1.1"
        
        print(f"签名原始字符串:\n{repr(signature_origin)}")
        
        # 使用HMAC-SHA256计算签名
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")
        
        print(f"计算出的签名: {signature_sha}")
        
        # 构建authorization字符串
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )
        
        print(f"Authorization原始字符串: {authorization_origin}")
        print(f"Authorization编码后: {authorization}")
        
        params = {
            "authorization": authorization,
            "date": date,
            "host": "iat.cn-huabei-1.xf-yun.com",
        }
        
        return url + "?" + urlencode(params)

    def on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            message = json.loads(message)
            code = message["header"]["code"]

            if code != 0:
                self.error_message = (
                    f"API错误码: {code}, 消息: {message['header'].get('message', '')}"
                )
            else:
                self.test_result = "连接成功！API配置正确。"

            self.is_finished = True
            ws.close()

        except Exception as e:
            self.error_message = f"解析消息失败: {e}"
            self.is_finished = True

    def on_error(self, ws, error):
        """处理WebSocket错误"""
        self.error_message = f"WebSocket错误: {error}"
        self.is_finished = True

    def on_close(self, ws, close_status_code, close_msg):
        """处理WebSocket关闭"""
        if not self.is_finished:
            self.error_message = f"连接意外关闭: {close_status_code} - {close_msg}"
        self.is_finished = True

    def on_open(self, ws):
        """处理WebSocket连接建立"""
        def run(*args):
            try:
                # 发送测试数据
                test_data = {
                    "header": {"app_id": self.app_id, "status": 0},
                    "parameter": {
                        "iat": {
                            "language": "zh_cn",
                            "accent": "mulacc",
                            "domain": "slm",
                            "result": {"encoding": "utf8", "compress": "raw", "format": "json"},
                        }
                    },
                    "payload": {
                        "audio": {
                            "encoding": "raw",
                            "sample_rate": 16000,
                            "channels": 1,
                            "bit_depth": 16,
                            "status": 0,
                            "seq": 0,
                            "audio": "",  # 空音频数据用于测试
                        }
                    },
                }

                ws.send(json.dumps(test_data))

                # 发送结束帧
                end_data = {
                    "header": {"app_id": self.app_id, "status": 2},
                    "payload": {
                        "audio": {
                            "encoding": "raw",
                            "sample_rate": 16000,
                            "channels": 1,
                            "bit_depth": 16,
                            "status": 2,
                            "seq": 0,
                            "audio": "",
                        }
                    },
                }

                ws.send(json.dumps(end_data))

            except Exception as e:
                self.error_message = f"发送测试数据失败: {e}"
                self.is_finished = True

        threading.Thread(target=run).start()

    def test_connection(self):
        """测试API连接"""
        print("正在测试讯飞API连接...")

        try:
            ws_url = self.create_url()
            print(f"WebSocket URL: {ws_url}")

            ws = websocket.WebSocketApp(
                ws_url, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close
            )
            ws.on_open = self.on_open

            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

            # 等待测试完成
            timeout = 10
            start_time = time.time()
            while not self.is_finished and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self.is_finished:
                return False, "测试超时"

            if self.error_message:
                return False, self.error_message
            else:
                return True, self.test_result

        except Exception as e:
            return False, f"连接失败: {e}"


def main():
    """主函数"""
    print("讯飞API配置测试工具")
    print("=" * 50)

    # 获取用户输入
    app_id = input("请输入APPID: ").strip()
    api_key = input("请输入APIKey: ").strip()
    api_secret = input("请输入APISecret: ").strip()

    if not app_id or not api_key or not api_secret:
        print("错误: 请提供完整的API配置信息")
        return

    # 创建测试实例
    tester = SparkAPITest(app_id, api_key, api_secret)

    # 执行测试
    success, message = tester.test_connection()

    print("-" * 50)
    if success:
        print(f"✓ 测试成功: {message}")
        print("您可以在 spark_iat_recognition.py 中使用这些配置。")
    else:
        print(f"✗ 测试失败: {message}")
        print("请检查您的API配置是否正确。")

    print("注意事项:")
    print("1. 确保您的网络连接正常")
    print("2. 检查API密钥是否正确")
    print("3. 确认您的讯飞账户有足够的余额")
    print("4. 检查应用是否已添加语音听写服务")


if __name__ == "__main__":
    main()
