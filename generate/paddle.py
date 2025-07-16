from paddlespeech.cli.tts import TTSExecutor

tts_executor = TTSExecutor()
# for i in range(0, 1):
#     wav_file = tts_executor(
#         text="你今日食咗饭未呀？我哋一齐去茶餐厅食个午餐啦！",
#         am="fastspeech2_canton",
#         voc="hifigan_csmsc",
#         lang="canton",
#         spk_id=i,
#         use_onnx=True,
#         output=f"id_{i}.wav",
#         cpu_threads=2,
#     )

# 6 和 8 发音不错
wav_file = tts_executor(
    text="呢套戏好好睇，如果你得闲嘅话，我哋可以一齐去戏院睇.  ",
    am="fastspeech2_canton",
    voc="hifigan_csmsc",
    lang="canton",
    spk_id=6,
    use_onnx=True,
    output=f"id_{6}-1.wav",
    cpu_threads=2,
)

wav_file = tts_executor(
    text="呢套戏好好睇，如果你得闲嘅话，我哋可以一齐去戏院睇.  ",
    am="fastspeech2_canton",
    voc="hifigan_csmsc",
    lang="canton",
    spk_id=8,
    use_onnx=True,
    output=f"id_{8}-1.wav",
    cpu_threads=2,
)