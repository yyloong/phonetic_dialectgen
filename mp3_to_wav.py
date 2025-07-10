import multiprocessing as mp
from tqdm import tqdm
import os
import subprocess


def process_file(mp3_file):
    wav_file = os.path.join(
        os.path.dirname(mp3_file) + "_wav",
        os.path.basename(mp3_file).replace(".mp3", ".wav"),
    )

    command = ["ffmpeg", "-i", mp3_file, wav_file]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {mp3_file}, 错误: {e}")


def parallel_process(mp3_dir, num_workers=mp.cpu_count()):
    os.makedirs(mp3_dir + "_wav", exist_ok=True)
    with mp.Pool(processes=num_workers) as pool:
        mp3_file_list = [
            os.path.join(mp3_dir, name)
            for name in os.listdir(mp3_dir)
            if name.endswith(".mp3")
        ]
        results = list(
            tqdm(
                pool.imap(
                    process_file,
                    mp3_file_list,
                ),
                total=len(mp3_file_list),
                desc="mp3 to wav files",
            )
        )
    return results


paths = [
    #"/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/nan-tw/clips",
    #"/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/yue/clips",
    "/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/zh-HK/clips",
    "/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/zh-TW/clips",
]
for path in paths:
    parallel_process(path)
