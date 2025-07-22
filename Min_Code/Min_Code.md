#### Min_Code
Min_Code 是能够复现模型的最小代码
#### 代码框架说明
- Character_to_IPA
这个文件夹里面Text_to_IPA.py 中包含将文本转化为IPA的接口,Text_to_IPA_in_csv.py 调用这一接口对.csv文件进行处理

- Wav_to_Mel
这个文件j夹面wav_to_melspec.py 中包含将波形图转化为梅尔频谱的接口
parallel_wav_to_melspec.py 中提供了更加具体的使用以及通过并行对某个目录下的所有.wav文件进行统一处理为.pt文件的接口

- layers
模型定义所需的网络模块

- config.py
用于对模型参数和训练参数进行配置

- dataset.py
定义了模型需要的特定数据集类和相应的collate_fn方法

- model.py
定义了模型结构的类

- synthesize.py
用于加载训练好的模型权重并进行输入输出测试

- tokenizer.py
用于将模型文本入转化为IPA

- trainer.py
定义了模型训练的类

- train.py
定义了模型训练的代码

- utils.py
定义了一些工具方法

#### 使用说明
请从XXXXXXX 下载数据集，并利用上述API对文本数据和.wav文件进行处理得到IPA和梅尔频谱的.pt文件,然后将文本数据整理到一个.csv文件中,确保audio这一列和数据存放地址对应(不含后缀名),然后在config.py文件中指定存放梅尔频谱的root_path和csv文件的path,通过train.py对模型进行训练(需要指定是否通过checkpoint进行训练)并保存checkpoint,在synthesize.py中运行模型进行测试