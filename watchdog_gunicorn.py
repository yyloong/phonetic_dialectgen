import subprocess
import time
import os
import signal

GUNICORN_CMD = ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "web:app", "--timeout", "60"]

def is_process_running(proc):
    return proc.poll() is None

def start_gunicorn():
    print("启动 gunicorn ...")
    return subprocess.Popen(GUNICORN_CMD)

def main():
    proc = start_gunicorn()
    try:
        while True:
            time.sleep(5)
            if not is_process_running(proc):
                print("检测到 gunicorn 进程已退出，正在重启 ...")
                proc = start_gunicorn()
    except KeyboardInterrupt:
        print("收到中断信号，关闭 gunicorn ...")
        if is_process_running(proc):
            os.kill(proc.pid, signal.SIGTERM)
        print("已退出。")

if __name__ == "__main__":
    main()