# 制作方言数据集

## 下载抖音视频

下载示例:

- 工具：`f2 douyin`，[Github:Johnserf-Seed/TikTokDownload](https://github.com/Johnserf-Seed/TikTokDownload)
- 抖音用户：[抖音:jiangtonghe3](https://www.douyin.com/user/MS4wLjABAAAAU6P9KQRWimKhmZdieoaU10RczgXsVhFWyXh7AylE150)
- 从视频中提取音频，例如`/misc/mp4_to_wav.py`

注意排除缺少真实语音的视频。（例如：特效视频、对口型视频等等。）

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 1. 音量归一化
python normalize_audio.py

# 2. 音频分割
python split_audio.py
```

## 文件结构

```
/wavs/           # 原始音频文件
/balanced/       # 归一化后的音频文件
/shorts/         # 分割后的短音频文件
/shorts_noisy/   # 重试失败文件的输出目录
```

## 失败文件重试

1. 首次运行后，自动生成 `split_audio_config.json`
2. 重新运行 `split_audio.py`，失败文件会重试并输出到 `shorts_noisy/`
3. 可手动编辑配置文件调整参数

配置文件示例：

```json
{
  "failed_files": [
    "problem_file1.wav",
    "problem_file2.wav"
  ],
  "relaxed_thresholds": {
    "silence_thresh": -40,
    "min_silence_len": 300,
    "target_length": 8000,
    "force_split": true
  }
}
```
