### Min_Code环境配置与结构说明
#### 环境配置
**请确保以下所有命令均在Min_Code目录下进行**

##### 安装python库依赖
conda install python=3.13
pip install -r requirements.txt

##### 从huggingface上下载bigvgan模型b并且从NJU-box下载训练数据并进行解压
- download_data.sh中第3行的download_wav决定了下载哪一类型的数据,True代表下载音频,否则下载梅尔频谱图
chmod +x download_data.sh
./download_data.sh
如果自行下载需要注意放置在和配置文件对应的位置

如果下载.wav数据,可以通过执行下载的命令来处理为.pt文件:
chmod +x auto_wav_to_mel.sh
./auto_wav_to_mel.sh
如果自行处理可以调用Wav_to_Mel中的接口进行，但是要注意生成文件路径要和.csv文件中对应

编译 Vits/monotonic_align/core.pyx 
python Vits/monotonic_align/setup.py build_ext --inplace

#### 配置文件格式说明
- 采用.toml格式,放置在Tomls_config目录下,目前有Vits.toml和Glow_TTS.tom两种模型的配置参数
- [model] 中定义了整个模型共用的参数和各个组件的参数
- [train] 中定义了训练相关的参数,包括epoch:训练轮数,print_step:多少步打印一次信息(每运行一个batch为一步),save_step:多少步保存一次,use_scheduler:是否使用调度器,logs_step:多少步记录一次tensorboard日志以及各种文件的路径,其中mandarin.csv和cantonese.csv分开存储但是一起训练,_num可以指定两种数据参与训练的数量，默认为全部数据
- [dataloader] [oprimizer] [scheduler] 的配置参数

#### 运行训练代码
python train_your_TTS.py

有3个可选参数:
- --config_path : 配置文件的路径,默认为Tomls_config/Glow_TTS.toml
- --model : 选择的模型架构,目前支持Vits和Glow-TTS两种模型,默认为Glow-TTS
- --checkpoint_path : 选择检查点的位置，默认为None,表示从头开始训练，否则加载检查点后训练

模型的权重和各种信息会根据配置文件的内容进行保存或者输出

在Min_Code目录下执行tensorboard --logdir xxx_logs 可以查看学习率和各项损失(本地电脑通过ssh访问服务器上的日志地址要先进行端口映射)

- 关于训练次数,在给定参数下,Glow-TTS训练40-80个epoch可以进行生成但是有杂音,更好的效果需要更多训练,每次训练终止可以通过加载检查点来继续训练,Vits训练20-30给epoch可以进行生成但是有杂音

- 除了命令行输入参数也可以在main中修改默认值来设置参数

#### 运行TTS任务
python start_your_TTS.py 
- --config_path 配置文件路径,默认为Tomls_config/Glow_TTS.toml
- --text 输入的文本
- --language 希望生成的语音,目前支持pinyin和jyutping两种
- --output_path 生成音频文件的存放路径
- --checkpoint_path 加载模型的检查点位置(此处的检查点与release中的模型权重并不完全兼容,如果需要加载需要先创建config.py文件并写入以下内容
```python
class GlowTTSConfig:
    def __init__(self):
        pass
```
)
- 除了命令行输入参数也可以在main中修改默认值来设置参数

#### 代码结构
Min_Code/
├── Character_to_IPA/              # 提供文本到IPA转变的接口
├── from_IPA_Tensor/                # 提供IPA到tensor编码的接口(采用整数编码)
├── Glow_TTS/        # Glow_TTS的结构定义代码
├── Vits/           # Vits的结构定义代码
├── Tomls_config/         # 配置文件目录
├── Wav_to_Mel/             # 提供语音转为梅尔频谱的接口
├── dataset.py              # 定义了数据集类
├── download_data.sh    # 用于下载解压数据的shell命令
├── load_bigvgan.py            # 提供了使用bigvgan来生成语音的接口
├── load_config.py           # 提供了加载配置文件的接口
├── load_save_checkpoint.py          # 提供了加载和保存检查点的接口
├── requirements.txt              # python库依赖
├── Trainer.py            # 定义训练器类,提供了统一的加载配置文件的接口,Vits,Glow-TTs中的trainer是其子类
├── start_your_TTS.py       # 运行模型
├── train_your_TTS.py            # 训练接口