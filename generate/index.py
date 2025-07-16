from indextts.infer import IndexTTS
import csv

tts = IndexTTS(model_dir="/home/u-wuhc/index-tts/checkpoints",cfg_path="/home/u-wuhc/index-tts/checkpoints/config.yaml")
voice = "/home/u-wuhc/index-tts/audio/sbn.wav"

with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        i, text = row
        output_file = f"data/{i}.wav"
        tts.infer(voice, text, output_file)