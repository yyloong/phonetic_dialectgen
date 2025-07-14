from pypinyin import pinyin, lazy_pinyin, Style

print(lazy_pinyin("衣裳", style=Style.TONE3, neutral_tone_with_five=True))
