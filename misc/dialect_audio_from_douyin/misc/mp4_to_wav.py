import os
from pathlib import Path
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

src_dir = Path("./mp4s")
dst_dir = Path("./wavs")
dst_dir.mkdir(exist_ok=True)

mp4_files = list(src_dir.glob("*.mp4"))


def convert(mp4_file):
    """
    将mp4文件转换为wav文件

    PCM格式, 16000Hz, 16bit, 单声道

    注：上述格式可以直接接入讯飞方言大模型，详见 https://www.xfyun.cn/doc/spark/spark_slm_iat.html
    """
    base_name = mp4_file.stem[:19]
    wav_file = dst_dir / f"{base_name}.wav"
    cmd = [
        "ffmpeg",
        "-i",
        str(mp4_file),
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_file),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


if __name__ == "__main__":
    with Pool(4) as p:
        list(tqdm(p.imap_unordered(convert, mp4_files), total=len(mp4_files)))
