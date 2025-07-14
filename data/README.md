process Mel spectrgram and run model

# 梅尔频谱图的数据存储说明
## AItts 数据集
每个AItts[i]下面的data和melspec的文件内容是对应的
data中的10.wav对应melspec中的10.pt
## LibriSpeech
/mnt/nas/shared/datasets/voices/LibriSpeech/train-clean-360/7145/**87280_wav_melspec**/7145-87280-0048.**pt** 和 
/mnt/nas/shared/datasets/voices/LibriSpeech/train-clean-360/7145/**87280**/7145-87280-0048.**flac** 是对应的

## zhvoice 数据集
zhvoice/zhaidatatang/xxx/xxx.mp3
zhvoice/zhaidatatang_mel/xxx/xxx.wav
zhvoice/zhaidatatang_mel/xxx/xxx.pt对应



## cv-corpus 数据集
/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/nan-tw/clips/common_voice_nan-tw_32369513.mp3
和
/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/nan-tw/clips_wal_melspec/common_voice_nan-tw_32369513.pt对应

## KeSpeech 数据集
/mnt/nas/shared/datasets/voices/KeSpeech/Audio/1027476/phase1/1027476_057083c0.wav
和 /mnt/nas/shared/datasets/voices/KeSpeech/Audio/1027476/phase_mel1/1027476_057083c0.pt对应

## MDCC
/mnt/nas/shared/datasets/voices/MDCC/audio/447_1803291552_92718_792.32_794.78.wav
和
/mnt/nas/shared/datasets/voices/MDCC/melspec/447_1803291552_92718_792.32_794.78.pt对应

## word_shk_cantonese
/mnt/nas/shared/datasets/voices/wordshk_cantonese_speech/data/melspec/098599.pt
和
/mnt/nas/shared/datasets/voices/wordshk_cantonese_speech/data/wav_files/098599.wav
对应