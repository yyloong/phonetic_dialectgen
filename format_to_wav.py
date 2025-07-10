import os
import multiprocessing as mp
from tqdm import tqdm
import subprocess
input_dir = "flac_files"  # FLAC 文件所在目录
output_dir = "wav_files"  # WAV 文件输出目录
file_paths= []
root_dir = "/mnt/nas/shared/datasets/voices/LibriSpeech/train-clean-360"
def get_librispeech_paths(root_dir):
    for speaker_id in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            file_paths.append(chapter_path)
    print(file_paths)
    print(len(file_paths))
def process_file(flac_dir):
    wav_dir=flac_dir+"_wav"
    os.makedirs(wav_dir, exist_ok=True)
    flac_file_list=os.listdir(flac_dir)
    for flac_file_name in flac_file_list:
        if not flac_file_name.endswith(".flac"):
            continue
        wav_file = os.path.join(wav_dir, flac_file_name.replace(".flac", ".wav"))
        flac_file= os.path.join(flac_dir, flac_file_name)
        command = ["ffmpeg", "-i", flac_file, wav_file]
        try:
            subprocess.run(command, check=True)
            print(f"转换成功: {flac_file} -> {wav_file}")
        except subprocess.CalledProcessError as e:
            print(f"转换失败: {flac_file}, 错误: {e}")
def parallel_process(num_workers=mp.cpu_count()):
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(
                    process_file,
                    file_paths,
                ),
                total=len(file_paths),
                desc="flac to wav files",
            )
        )
    return results
parallel_process()
