import epitran  # 导入 epitran 库用于汉字转音标
import panphon  # 导入 panphon 库用于音标特征处理

# 初始化 epitran，指定中文（简体）
epi = epitran.Epitran("cmn-Hans")
# 初始化 panphon 的特征表
ft = panphon.FeatureTable()

text = "月亮偷偷给海鸥送去柠檬。"  # 待转写的中文文本
# 使用 epitran 进行音标（ASCII）转写
ascii_transcription = epi.transliterate(text)

# 使用 panphon 的旧版 API，将 ASCII 音标分割为音素片段
segments = ft.ipa_segs(ascii_transcription)
# 将音素片段拼接为完整的 IPA 转写
ipa_transcription = "".join(segments)

# 输出 ASCII 转写和 IPA 转写结果
print(f"ASCII 转写: {ascii_transcription}")
print(f"IPA 转写: {ipa_transcription}")
