import sys
sys.path.append("/home/u-wuhc/index-tts")
from indextts.infer import IndexTTS
import csv

tts = IndexTTS(model_dir="/home/u-wuhc/index-tts/checkpoints",cfg_path="/home/u-wuhc/index-tts/checkpoints/config.yaml")
voice = "/home/u-wuhc/index-tts/audio/sbn.wav"

tts.infer(voice, "暑假来了，你感受到预期的放松和快乐了吗？", "output.wav")

# with open('data.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         i, text = row
#         output_file = f"data/{i}.wav"
#         tts.infer(voice, text, output_file)