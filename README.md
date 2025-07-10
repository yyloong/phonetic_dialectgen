# week2
process Mel spectrgram and run model

# 梅尔频谱图的数据存储说明
## AItts 数据集
-- AItts
   -- AItts1
      --  data
      --  melspec
   -- AItts2
      ...
   -- AItts3
      ...
每个AItts[i]下面的data和melspec的文件内容是对应的,data中的10.wav对应melspec中的10.pt
## LibriSpeech
-- LibriSpeech
   -- train-clean-360
      -- 7145 
          -- 87280
          -- 87280_wav
          -- 87280_wav_melspec
      --  2890
        ...
train-clean-360中的每个文件夹的名称是一个数字,文件夹里面还有以数字命名的文件夹(如87280,里面是flac文件)
87280_wav是将.flac变成.wav格式后的文件夹
87280_wav_melspec 是进一步变成梅尔频谱的文件夹
比如 /mnt/nas/shared/datasets/voices/LibriSpeech/train-clean-360/7145/**87280_wav_melspec**/7145-87280-0048.**pt** 和 /mnt/nas/shared/datasets/voices/LibriSpeech/train-clean-360/7145/**87280**/7145-87280-0048.**flac** 是对应的

## zhvoice 数据集
zhaidatatang/xxx/xxx.mp3 zhaidatatang_mel/xxx/xxx.wav  和 zhaidatatang_mel/xxx/xxx.pt对应