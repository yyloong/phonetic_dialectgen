### Min_Code

Min_Code 是能够复现模型的最小代码集合。


#### 代码框架说明

| 模块/文件 | 功能说明 |
|-----------|----------|
| **Character_to_IPA** | 文本转IPA（国际音标）工具<br>- `Text_to_IPA.py`：提供文本转IPA的核心接口<br>- `Text_to_IPA_in_csv.py`：调用上述接口批量处理CSV文件 |
| **Wav_to_Mel** | 音频转梅尔频谱工具<br>- `wav_to_melspec.py`：提供波形图转梅尔频谱的核心接口<br>- `parallel_wav_to_melspec.py`：并行处理指定目录下的所有WAV文件，统一转为PT格式 |
| **layers** | 存放模型定义所需的网络模块组件 |
| **config.py** | 模型参数与训练参数的配置文件 |
| **dataset.py** | 定义模型所需的数据集类及对应的`collate_fn`方法 |
| **model.py** | 模型结构的核心类定义 |
| **synthesize.py** | 加载训练好的模型权重，用于输入输出测试 |
| **tokenizer.py** | 将模型的文本输入转换为IPA格式 |
| **trainer.py** | 模型训练逻辑的类定义 |
| **train.py** | 模型训练的执行代码（支持从checkpoint恢复训练） |
| **utils.py** | 通用工具方法集合 |


#### 使用说明

1. 从 [XXXXXXX](链接地址) 下载数据集
2. 使用工具预处理数据：
   - 文本数据：通过 `Character_to_IPA` 模块转换为IPA格式
   - 音频文件：通过 `Wav_to_Mel` 模块转换为梅尔频谱（PT格式）
3. 整理文本数据为CSV文件，确保 `audio` 列与梅尔频谱文件的存放路径对应（不含文件后缀）
4. 在 `config.py` 中配置：
   - `root_path`：梅尔频谱文件的存放根目录
   - `path`：CSV文件的路径
5. 训练模型：
   运行trainer.py
6. 测试模型：
   运行synthesize.py