# GlowTTS

type: Post
status: Draft
date: 2025/07/10

### 网络结构

- 代码
    
    ```
    GlowTTS(
      (encoder): Encoder(
        (emb): Embedding(129, 192)
        (prenet): ResidualConv1dLayerNormBlock(
          (conv_layers): ModuleList(
            (0-2): 3 x Conv1d(192, 192, kernel_size=(5,), stride=(1,), padding=(2,))
          )
          (norm_layers): ModuleList(
            (0-2): 3 x LayerNorm()
          )
          (proj): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
        )
        (encoder): RelativePositionTransformer(
          (dropout): Dropout(p=0.1, inplace=False)
          (attn_layers): ModuleList(
            (0-5): 6 x RelativePositionMultiHeadAttention(
              (conv_q): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              (conv_k): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              (conv_v): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              (conv_o): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (norm_layers_1): ModuleList(
            (0-5): 6 x LayerNorm()
          )
          (ffn_layers): ModuleList(
            (0-5): 6 x FeedForwardNetwork(
              (conv_1): Conv1d(192, 768, kernel_size=(3,), stride=(1,))
              (conv_2): Conv1d(768, 192, kernel_size=(3,), stride=(1,))
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (norm_layers_2): ModuleList(
            (0-5): 6 x LayerNorm()
          )
        )
        (proj_m): Conv1d(192, 80, kernel_size=(1,), stride=(1,))
        (duration_predictor): DurationPredictor(
          (drop): Dropout(p=0.1, inplace=False)
          (conv_1): Conv1d(192, 256, kernel_size=(3,), stride=(1,), padding=(1,))
          (norm_1): LayerNorm()
          (conv_2): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
          (norm_2): LayerNorm()
          (proj): Conv1d(256, 1, kernel_size=(1,), stride=(1,))
        )
      )
      (decoder): Decoder(
        (flows): ModuleList(
          (0): ActNorm()
          (1): InvConvNear()
          (2): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (3): ActNorm()
          (4): InvConvNear()
          (5): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (6): ActNorm()
          (7): InvConvNear()
          (8): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (9): ActNorm()
          (10): InvConvNear()
          (11): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (12): ActNorm()
          (13): InvConvNear()
          (14): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (15): ActNorm()
          (16): InvConvNear()
          (17): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (18): ActNorm()
          (19): InvConvNear()
          (20): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (21): ActNorm()
          (22): InvConvNear()
          (23): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (24): ActNorm()
          (25): InvConvNear()
          (26): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (27): ActNorm()
          (28): InvConvNear()
          (29): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (30): ActNorm()
          (31): InvConvNear()
          (32): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
          (33): ActNorm()
          (34): InvConvNear()
          (35): CouplingBlock(
            (start): ParametrizedConv1d(
              80, 192, kernel_size=(1,), stride=(1,)
              (parametrizations): ModuleDict(
                (weight): ParametrizationList(
                  (0): _WeightNorm()
                )
              )
            )
            (end): Conv1d(192, 160, kernel_size=(1,), stride=(1,))
            (wn): WN(
              (in_layers): ModuleList(
                (0-3): 4 x Conv1d(192, 384, kernel_size=(5,), stride=(1,), padding=(2,))
              )
              (res_skip_layers): ModuleList(
                (0-2): 3 x Conv1d(192, 384, kernel_size=(1,), stride=(1,))
                (3): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
              )
              (dropout): Dropout(p=0.05, inplace=False)
            )
          )
        )
      )
    )
    ```
    

参考的 CoquiTTS 对于 GlowTTS 的实现代码

以下是一些主要文件

/home/u-wuhc/.local/share/tts/

/home/u-wuhc/TTS/TTS/utils/synthesizer.py

/home/u-wuhc/TTS/TTS/utils/audio/processor.py

/home/u-wuhc/TTS/TTS/tts/models/glow_tts.py

/home/u-wuhc/TTS/TTS/tts/configs/glow_tts_config.py

/home/u-wuhc/TTS/recipes/ljspeech/glow_tts/train_glowtts.py

/home/u-wuhc/TTS/recipes/ljspeech/glow_tts/test.py

/home/u-wuhc/TTS/TTS/config/shared_configs.py

/home/u-wuhc/TTS/toMel.py

### TODO

- [ ]  Text Tokenizer
- [ ]  Losses 是什么
- [ ]  训练过程
- [ ]  Dataloader (train)
- [ ]  论文
- [ ]  load checkpoint
- [ ]  save model

- 如何处理长度不同的 x 输入
    
    x : text IDs
    
    在 Encoder 中处理
    
    ## GlowTTS 如何处理长度不同的 x 输入
    
    ### 🎯 **1. 批量处理不同长度的文本**
    
    GlowTTS 使用 **padding + mask** 的策略处理不同长度的文本输入：
    
    ```python
    # 示例：不同长度的文本序列
    batch = [
        "Hello world",      # 长度 = 11
        "Hi",              # 长度 = 2
        "How are you?"     # 长度 = 12
    ]
    
    # 转换为 token IDs 后：
    x = torch.tensor([
        [23, 15, 12, 12, 24, 0, 28, 24, 18, 12, 14, 0, 0],  # padding 到最大长度
        [23, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # padding 到最大长度
        [23, 24, 28, 0, 1, 18, 15, 0, 25, 24, 21, 0, 0]    # padding 到最大长度
    ])  # shape: [3, 13] - 批次大小=3, 最大长度=13
    
    x_lengths = torch.tensor([11, 2, 12])  # 各序列的实际长度
    
    ```
    
    ### 🎯 **2. 编码器中的 Mask 处理**
    
    ```python
    def forward(self, x, x_lengths, y, y_lengths=None):
        # 编码器处理文本，返回 mask
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=None)
        #                                                                    ↑
        #                                                              文本序列的 mask
    
        # x_mask 的作用：
        # x_mask[0] = [1,1,1,1,1,1,1,1,1,1,1,0,0]  # 前11个有效，后2个无效
        # x_mask[1] = [1,1,0,0,0,0,0,0,0,0,0,0,0]  # 前2个有效，后11个无效
        # x_mask[2] = [1,1,1,1,1,1,1,1,1,1,1,1,0]  # 前12个有效，后1个无效
    
    ```
    
    ### 🎯 **3. 参数 `num_chars` 的含义**
    
    `num_chars` 是 **词汇表大小**（vocabulary size），表示模型能识别的不同字符/token的数量：
    
    ```python
    config = GlowTTSConfig(num_chars=100, out_channels=80)
    #                      ↑
    #                  词汇表大小 = 100个不同的字符/token
    
    # 意味着：
    # - 文本会被转换为 0-99 范围内的整数 token
    # - 编码器的嵌入层维度为 [num_chars, embedding_dim]
    # - 例如：'a'→1, 'b'→2, 'c'→3, ..., ' '→0, '<PAD>'→99
    
    ```
    
    ### 🎯 **4. 编码器中的处理流程**
    
    ```python
    class Encoder:
        def __init__(self, num_chars, ...):
            # 字符嵌入层
            self.emb = nn.Embedding(num_chars, hidden_channels)
            #                       ↑           ↑
            #                  词汇表大小     嵌入维度
    
        def forward(self, x, x_lengths, g=None):
            # 1. 字符嵌入
            x = self.emb(x)  # [B, T_en] → [B, T_en, hidden_channels]
    
            # 2. 生成序列mask
            x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(1)), 1)
            #                                     ↑           ↑
            #                               实际长度     最大长度
    
            # 3. 应用mask到嵌入
            x = x * x_mask  # 将padding部分置零
    
            # 4. 后续的编码器层处理...
            return o_mean, o_log_scale, o_dur_log, x_mask
    
    ```
    
    ### 🎯 **5. 实际应用示例**
    
    ```python
    # 中文TTS的词汇表可能包含：
    vocabulary = {
        '<PAD>': 0,    # 填充token
        '<SOS>': 1,    # 开始token
        '<EOS>': 2,    # 结束token
        '你': 3,
        '好': 4,
        '世': 5,
        '界': 6,
        '，': 7,
        '。': 8,
        # ... 更多字符
    }
    
    # 如果词汇表有1000个字符，则：
    config = GlowTTSConfig(num_chars=1000, ...)
    
    # 文本 "你好世界" 会被转换为：
    text = "你好世界"
    tokens = [3, 4, 5, 6]  # 对应的token IDs
    x = torch.tensor([tokens])  # [1, 4]
    x_lengths = torch.tensor([4])  # 实际长度
    
    ```
    
    ### 🎯 **6. 长度处理的完整流程**
    
    ```python
    def process_batch(texts):
        # 1. 文本转token
        tokens_list = [text_to_tokens(text) for text in texts]
    
        # 2. 获取长度信息
        lengths = [len(tokens) for tokens in tokens_list]
        max_length = max(lengths)
    
        # 3. Padding到统一长度
        padded_tokens = []
        for tokens in tokens_list:
            padded = tokens + [0] * (max_length - len(tokens))  # 用0填充
            padded_tokens.append(padded)
    
        # 4. 转换为张量
        x = torch.tensor(padded_tokens)  # [B, max_length]
        x_lengths = torch.tensor(lengths)  # [B]
    
        return x, x_lengths
    
    ```
    
    ### 🎯 **总结**
    
    - **`num_chars`**: 词汇表大小，定义了模型能处理的不同字符/token数量
    - **长度处理**: 通过padding统一批次内的序列长度，用mask标记有效部分
    - **核心思想**: 物理上统一长度（便于并行计算），逻辑上保持原始长度（确保正确处理）
    
    这种设计使得GlowTTS能够高效地处理任意长度的文本输入！
    
- forward 和 inference 函数的区别
    
    forward 用于训练
    
    inference 用于推理
    
    y_lengths 是 mel-spectrogram 的真实长度
    
    ### 🎯 **关键区别**：
    
    - **`y_lengths`** 每个样本的**实际有效长度**
    - **`T_de`** 批次中的**最大长度**（用于张量填充）
    
    ### 🔍 **详细解释**：
    
    ```python
    # 假设有一个批次包含3个样本：
    batch = [
        sample1: 实际长度 = 80 帧
        sample2: 实际长度 = 120 帧
        sample3: 实际长度 = 100 帧
    ]
    
    # 那么：
    y_lengths = torch.tensor([80, 120, 100])  # 各样本的实际长度
    T_de = 120  # 批次中的最大长度，用于张量填充
    
    # 最终的张量形状：
    y.shape = [3, 120, 80]  # [B, T_de, C_mel]
    #           ↑   ↑    ↑
    #           |   |    梅尔频谱维度
    #           |   最大长度（填充后）
    #           批次大小
    
    ```
    
    ### 🎯 **在 forward() 中的处理**：
    
    ```python
    def forward(self, x, x_lengths, y, y_lengths=None):
        # 输入：y.shape = [B, T_de, C_mel]
        # 输入：y_lengths = [实际长度1, 实际长度2, ...]
    
        # 1. 转换维度
        y = y.transpose(1, 2)  # [B, C_mel, T_de]
        y_max_length = y.size(2)  # T_de = 最大长度
    
        # 2. 预处理（可能会调整长度）
        y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
    
        # 3. 创建掩码（关键！）
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        #                                     ↑           ↑
        #                              实际长度    最大长度
    
    ```
    
    ### 🎯 **为什么需要 `y_lengths`？**
    
    ```python
    # 示例：sequence_mask 的作用
    y_lengths = torch.tensor([80, 120, 100])
    y_max_length = 120
    
    mask = sequence_mask(y_lengths, y_max_length)
    # 结果：
    # mask[0] = [1,1,1,...,1,0,0,...,0]  # 前80个为1，后40个为0
    #            ←—— 80 ——→ ←—— 40 ——→
    # mask[1] = [1,1,1,...,1]            # 全120个为1
    # mask[2] = [1,1,1,...,1,0,0,...,0]  # 前100个为1，后20个为0
    #            ←—— 100 ——→ ←—— 20 ——→
    
    ```
    
    ### 🎯 **总结**：
    
    - **`T_de`**: 张量的物理维度（最大长度）
    - **`y_lengths`**: 每个样本的逻辑长度（实际有效长度）
    - **作用**: `y_lengths` 用于生成掩码，确保模型只处理有效数据，忽略填充部分
    
    所以 `y_lengths` 不是 `T_de` 的值，而是每个样本在 `T_de` 维度上的**有效长度**！
    

### inference 过程

![image.png](GlowTTS%2022cb1baf8a678027a798c293618a8f5a/image.png)

**文本 -> 编码器 -> (均值μ, 时长d) -> 扩展对齐 -> 采样z -> 解码器 -> 梅尔频谱图**

- 主要流程
    
    我们可以参考论文中的 **Figure 1(b) An abstract diagram of the inference procedure** 来逐步分解。
    
    ### 核心思想
    
    Glow-TTS在前向推理时，完全是 **非自回归（non-autoregressive）**的。这意味着它一次性生成整个梅尔频谱图，而不是像Tacotron 2那样一帧一帧地生成。这个过程不依赖于前一时刻的输出，因此可以完全并行化，速度极快。
    
    最重要的一点是，在训练阶段使用的 **MAS（Monotonic Alignment Search）算法在推理时完全不使用**。MAS的作用是为训练找到文本和语音之间最可能的对齐关系，并用这个关系来训练**时长预测器（Duration Predictor）**。一旦时长预测器训练好了，推理时就只依赖它来确定对齐。
    
    ### 推理流程详解 (Step-by-Step)
    
    ### 步骤 1: 文本编码 (Encoder)
    
    1. **输入**: 一段文本序列，通常会预处理成音素（phonemes）序列，例如 "hello world" -> `[h, ə, l, oʊ, w, ɜː, l, d]`。然后通过 tokenizer 将每个处理为一个整数。
    2. **组件**: **文本编码器 (Encoder)**，它是一个基于Transformer的结构。
    3. **操作**: 编码器接收音素序列，并通过多层自注意力（self-attention）和前馈网络进行处理。
    4. **输出**:
        - **隐藏表示 (Hidden Representations)**: 为每个输入的音素生成一个隐藏状态向量 `h`。
        - **先验分布参数 (Prior Distribution Statistics)**: 从隐藏状态`h`通过一个线性层（图中的`Project`）预测出先验分布（一个高斯分布）的**均值 `μ` 和标准差 `σ`**。所以，对于每个输入音素，我们都有对应的 `μ_i` 和 `σ_i`。
    
    ### 步骤 2: 时长预测 (Duration Predictor)
    
    1. **输入**: 文本编码器输出的隐藏表示 `h`。（注意：论文提到在训练时长预测器时，会使用`stop_gradient`来防止其梯度影响编码器的训练，但这在推理时没有影响）。
    2. **组件**: **时长预测器 (Duration Predictor)**。
    3. **操作**: 该模块预测每个输入音素应该持续多长时间，即对应多少个梅尔频谱图的帧（frames）。
    4. **输出**: 一个时长序列 `d = (d_1, d_2, ..., d_T_text)`。`d_i` 是一个浮点数，表示第`i`个音素对应的时长。
    
    ### 步骤 3: 对齐扩展/上采样 (Alignment Expansion)
    
    这是将文本域的信息映射到语音域的关键一步，也是取代训练时MAS算法的一步。
    
    1. **输入**:
        - 来自**步骤1**的先验分布参数序列 `(μ_1, σ_1), (μ_2, σ_2), ...`
        - 来自**步骤2**的预测时长序列 `d = (d_1, d_2, ...)`
    2. **操作**:
        - 首先，将浮点数的时长`d_i`转换为整数。在Figure 1(b)中，这一步被标记为 `Ceil`，表示向上取整。这确保了每个音素至少对应一帧，避免了信息丢失。
        - 然后，根据整数时长 `d'_i = ceil(d_i)`，将每个音素对应的 `μ_i` 和 `σ_i` **重复** `d'_i` 次。
        - 例如，如果音素 "h" 的预测时长是 `2.3`（向上取整为3），那么就将 "h" 对应的 `μ_h` 和 `σ_h` 复制3遍。
    3. **输出**: 两个被"拉伸"或"扩展"了的序列，`μ_expanded` 和 `σ_expanded`。它们的总长度 `T_mel` 等于所有音素时长的总和（`T_mel = Σ d'_i`），这个长度就是最终要生成的梅尔频谱图的时间步长。
    
    ### 步骤 4: 采样生成隐变量 (Latent Variable Generation)
    
    1. **输入**: 扩展后的均值序列 `μ_expanded`。
    2. **操作**: 从先验分布中采样生成隐变量 `z`。这个过程非常简单：
        - 首先，生成一个与`μ_expanded`同样大小的随机噪声 `ε`，该噪声从标准正态分布 `N(0, I)` 中采样。
        - 然后，根据公式 `z = μ + ε * T` 计算隐变量 `z`。
            - `μ` 就是 `μ_expanded`。
            - `ε` 是随机噪声。
            - `T` 是一个温度（temperature）超参数。在推理时，通过调整`T`可以控制生成语音的多样性。`T` 越小，语音越接近均值，变化越少；`T` 越大，随机性越强，韵律（prosody）变化越丰富。
    3. **输出**: 一个隐变量张量 `z`，其维度与最终的梅-尔频谱图相同。
    
    ### 步骤 5: 并行解码 (Flow-based Decoder)
    
    1. **输入**: 步骤4中生成的隐变量 `z`。
    2. **组件**: **基于流的解码器 (Flow-based Decoder)**，它是一系列可逆的转换（Invertible Transforms），如ActNorm、Invertible 1x1 Conv和Affine Coupling Layer。
    3. **操作**: 解码器执行**逆向转换（inverse transformation）**。在训练时，解码器学习将真实的梅尔频谱图 `x` 映射到隐变量 `z`（`x -> z`）。在推理时，它执行相反的操作，将采样的隐变量 `z` 映射回梅尔频谱图 `x`（`z -> x`）。这个转换过程的每一步都是并行计算的。
    4. **输出**: 最终的**梅尔频谱图 (Mel-Spectrogram)**。
    
    ### 总结
    
    Glow-TTS的推理流程可以概括为：
    **文本 -> 编码器 -> (均值μ, 时长d) -> 扩展对齐 -> 采样z -> 解码器 -> 梅尔频谱图**
    
    这个流程的**优点**非常突出：
    
    - **快速**: 整个过程没有循环依赖，可以完全并行化，推理速度极快，几乎与输入文本长度无关（只与最长文本有关）。
    - **鲁棒**: 由于使用了硬对齐（由时长预测器决定），它不会像基于注意力机制的自回归模型那样在处理长文本或重复词时出现注意力错误（如跳词、重复发音）。
    - **可控**:
        - 通过调整**时长预测器**的输出（例如，乘以一个系数），可以轻松控制语速。
        - 通过调整**温度`T`**，可以控制生成语音的韵律和音调变化。

### Encoder

```
Glow-TTS encoder module.:

    embedding -> <prenet> -> encoder_module -> <postnet> --> h (隐藏表示序列) proj_mean
			                                                        |
			                                                        |-> <postnet> -> proj_mean / proj_var
			                                                        |
			                                                        |-> concat -> duration_predictor
		                                                                ↑
		                                                          speaker_embed
```

- 主要流程
    
    这个编码器模块的核心任务是：**接收输入的文本（音素序列），并将其转换为一个富含信息的隐藏表示，这个表示既包含了语音的“内容”信息，也包含了语音的“时长”信息。** 它的输出是后续解码步骤的基石。
    
    下面我们按照流程图的顺序一步步解析：
    
    ---
    
    ### 1. 输入与嵌入 (Input & Embedding)
    
    - **输入 (Input):** 编码器的输入不是原始的文字（如"hello"），而是经过预处理的**音素序列 (phoneme sequence)**。例如，文本"Glow-TTS"会被转换成一串音素ID，如 `[jh, l, ow, t, iy, t, iy, eh, s]`。使用音素可以更好地处理拼写不规则的单词，使模型学习更加稳定。
    - **嵌入层 (`embedding`):** 这是一个标准的嵌入层。它将输入的离散音素ID（比如整数 `[45, 51, 64, ...]`）映射成连续的、固定维度的向量。这个向量就是音素的初始表示。
    
    ### 2. 前置网络 (Pre-net)
    
    - **组件 (`<prenet>`):** 在进入核心的Transformer模块之前，嵌入向量会先经过一个“前置网络”。
    - **作用:** Pre-net通常由几层卷积层或全连接层构成，并带有非线性激活函数（如ReLU）和Dropout。它的主要作用是对嵌入向量进行初步的特征提取和变换，增加模型的非线性能力和鲁棒性，为后续的复杂处理（自注意力）准备一个更好的输入表示。
    
    ### 3. 核心编码模块 (Encoder Module)
    
    - **组件 (`encoder_module`):** 这是编码器的心脏，在Glow-TTS中，它是一个基于**Transformer**的编码器。它由多个相同的块堆叠而成。
    - **作用:** 它的核心是**自注意力机制 (Self-Attention)**。对于序列中的每一个音素，自注意力机制会计算它与序列中所有其他音素的关联度（或称“注意力权重”）。这使得模型能够捕捉长距离的依赖关系和上下文信息。例如，模型可以理解到，一个音素的发音会受到其前后音素的影响。
    - **输出:** 经过这个模块处理后，我们得到一个隐藏表示序列 `h`。序列中的每一个向量 `h_i` 都编码了第 `i` 个音素及其上下文的丰富语言学信息。
    
    ### 4. 输出生成 (Output Generation)
    
    从核心编码模块输出的隐藏表示 `h` 会被用于生成两个关键的输出：
    
    **4.1 先验分布的参数 (Prior Distribution Parameters)**
    
    - **路径:** `h -> <postnet> -> proj_mean / proj_var`
    - **解释:** 这条路径负责生成语音的“内容”。
        - **Post-net (`<postnet>`):** 这里的`<postnet>`可以理解为最后的线性投影层。
        - **投影 (`proj_mean`, `proj_var`):** 隐藏表示 `h` 通过两个独立的线性层，分别预测出先验高斯分布的**均值 (mean `μ`)** 和 **标准差 (standard deviation `σ`)**。（流程图中的`proj_var`通常在实现时是预测对数标准差 `log σ`，以保证其为正并增加训练稳定性）。
        - **结果:** 对于输入的每一个音素，我们都得到了一个对应的 `μ_i` 和 `σ_i`。这两个参数定义了在生成语音时，该音素应该从哪个高斯分布中采样。`μ` 决定了音色的基本特征。
    
    **4.2 音素时长 (Phoneme Duration)**
    
    - **路径:** `h -> concat(h, speaker_embed) -> duration_predictor`
    - **解释:** 这条路径负责生成语音的“时长”**或**“节奏”。
        - **说话人嵌入 (`speaker_embed`):** 在**多说话人（multi-speaker）模型中，为了让模型知道要用哪个人的语速来说话，隐藏表示 `h` 会与一个说话人嵌入向量 (speaker embedding)** 进行拼接（`concat`）。这个嵌入向量代表了特定说话人的身份特征（包括平均语速）。在单说话人模型中，这一步可以省略。
        - **时长预测器 (`duration_predictor`):** 拼接后的向量被送入时长预测器。这是一个相对简单的网络（通常是几层卷积层），它的任务是为每一个音素预测一个标量值，即该音素应该持续的**时长**（对应多少个梅尔频谱帧）。
        - **结果:** 我们得到了一个时长序列 `d`，其中 `d_i` 对应第 `i` 个音素的预测时长。
    
    ---
    
    ### 总结
    
    Glow-TTS的编码器设计非常精巧，它将复杂的语音生成任务解耦为两个子问题：
    
    1. **“说什么内容？”** —— 由 `μ` 和 `σ` 决定，它们定义了每个音素的声学特征。
    2. **“说多长时间？”** —— 由 **时长预测器** 决定，它确定了每个音素的持续时间。
    
    这种设计使得Glow-TTS在推理时可以先通过**时长预测器**确定好完整的对齐关系，然后**一次性、并行地**从先验分布中采样并解码出整个梅尔频谱图，从而实现了极快的合成速度和高度的鲁棒性。
    

### MAS

pass

### Loss

好的，Glow-TTS在训练时的损失函数（loss）由两个主要部分组成，分别对应模型要学习的两个核心任务：**生成正确的声学特征**和**预测正确的音素时长**。

这两个损失分别是：

1. **最大似然损失 (Maximum Likelihood Loss)**
2. **时长预测损失 (Duration Prediction Loss)**
- 详细解析这两个损失。
    
    ### 1. 最大似然损失 (Maximum Likelihood Loss)
    
    这是Glow-TTS模型最核心的损失，用于训练**文本编码器（Encoder）和基于流的解码器（Flow-based Decoder）**。
    
    **目标：** 最大化给定文本条件 c 下，模型生成真实梅尔频谱图 x 的对数似然概率 log P(x|c)。
    
    **原理：**
    
    Glow-TTS是一个基于流的生成模型，它利用了**变量代换公式（Change of Variables Formula）**。解码器 f_dec 是一个可逆函数，可以将简单的先验分布（如标准正态分布）中的隐变量 z 映射到复杂的数据分布（梅尔频谱图 x）。
    
    其对数似然可以表示为：
    
    $$
    \log P_X(x|c) = \log P_Z(z|c) + \log \left|\det\frac{\partial f^{-1}_{dec}(x)}{\partial x}\right|
    $$
    
    在训练时，我们是反向计算，从 $x$ 映射到 $z$（$z = f_{dec}^{-1}(x)$）：
    
    $$
    \log P_X(x|c) = \log P_Z(f_{dec}^{-1}(x)|c) + \log \left|\det\frac{\partial f^{-1}_{dec}(x)}{\partial x}\right|
    $$
    
    这个公式包含两项：
    
    - ****$\log P_Z(f_{dec}^{-1}(x)|c)$: 这一项是**先验分布的对数似然**。
        - 首先，通过解码器 f_dec 的逆向传播，将真实的梅尔频谱图 x 转换成隐变量 z。
        - 然后，通过编码器 f_enc 得到先验分布的参数 μ 和 σ。
        - 最关键的一步是，**通过独热对齐搜索（MAS）算法找到 z 和 (μ, σ) 之间最可能的对齐关系 A***。
        - 最后，计算在对齐关系 A* 下，z 服从以 μ 和 σ 为参数的高斯分布的对数概率。这部分损失会驱动编码器和解码器学习生成正确的声学内容。
    - $\log \left|\det\frac{\partial f^{-1}_{dec}(x)}{\partial x}\right|$: 这一项是**雅可比行列式的对数值（Log-determinant of Jacobian）**。
        - 它衡量了从 x 到 z 的变换过程中空间的缩放程度。
        - 对于Glow中使用的流模型组件（如affine coupling, invertible 1x1 conv），这一项可以被高效地计算出来。
    
    **最终的似然损失函数 L_mle 就是最大化这个对数似然，等价于最小化其负值：**
    
    $$
    L_{mle} = - \log P_X(x|c)
    $$
    
    ### 2. 时长预测损失 (Duration Prediction Loss)
    
    这个损失用于训练**时长预测器（Duration Predictor）**。
    
    **目标：** 让时长预测器预测出的音素时长，尽可能地接近由**独热对齐搜索（MAS）算法**找到的“真实”时长。
    
    **原理：**
    
    1. **获取真实时长 d**: 在每个训练步中，MAS算法会为当前的文本和语音对找到一个最佳的独热对齐 $A^*$。通过统计每个音素在该对齐中被分配了多少个梅尔频谱帧，就可以得到一个“真实”的时长序列 d。
        
        $$
        d_i = \sum_j^{T_{mel}} 1_{A^*(j)=i} \qquad (eq.5)
        $$
        
    2. **获取预测时长 d_hat**: 时长预测器接收编码器的隐藏表示 h，输出预测的时长序列 d_hat。
        
        d_hat = f_dur(sg[f_enc(c)]) (其中sg是stop-gradient)
        
    3. **计算损失 L_dur**: 使用均方误差（Mean Squared Error, MSE）来计算预测时长和真实时长之间的差距。通常这个计算是在对数域（log domain）进行的，因为时长是正数且分布可能很广，在对数域计算可以使训练更稳定。
        
        L_dur = MSE(log(d_hat), log(d)) (论文中公式6的简化形式)
        
    
    **特别注意 stop_gradient 操作:**
    
    在计算 L_dur 时，时长预测器的输入 f_enc(c) 被stop_gradient包裹。这意味着 L_dur 的梯度**不会**反向传播到文本编码器 f_enc。这样做是为了**解耦**两个学习任务：
    
    - 编码器的主要任务是学习声学内容，由 L_mle 驱动。
    - 时长预测器的任务是学习节奏，由 L_dur 驱动。
        
        如果不加 stop_gradient，时长预测器的误差可能会“污染”编码器，使其为了降低时长预测误差而改变其本应学习的声学表示，导致性能下降。
        
    
    ---
    
    ### 总损失 (Total Loss)
    
    最终，Glow-TTS的总损失是这两个损失的简单相加（或者加权相加，但论文中似乎是直接相加）：
    
    ```
    L_total = L_mle + L_dur
    ```
    
    通过同时优化这两个损失，Glow-TTS模型能够在一个端到端的框架中，既学习到如何生成高质量的语音内容，又学习到如何准确地控制语音的节奏和时长，并且这个过程完全不需要外部对齐工具的预处理。
    

### Train

![image.png](GlowTTS%2022cb1baf8a678027a798c293618a8f5a/image%201.png)

- 训练流程
    
    好的，我们来详细梳理一下Glow-TTS的完整训练过程。这个过程是其能够摆脱外部对齐器、实现端到端训练的关键。
    
    我们可以将整个训练过程分解为在一个训练批次（batch）内执行的几个连续步骤。参考论文中的 **Figure 1(a) An abstract diagram of the training procedure** 会很有帮助。
    
    ### 训练流程 (Step-by-Step for a Single Batch)
    
    假设我们有一个批次的数据，每条数据包含一对 **(文本序列 `c`, 梅尔频谱图 `x`)**。
    
    ### 步骤 1: 正向传播 (Forward Pass)
    
    1. **文本编码器 (Encoder) 部分:**
        - 将文本序列 `c`（音素）输入**文本编码器** (`f_enc`)。
        - 编码器输出两个结果：
            - **隐藏表示 `h`**: 用于后续的时长预测。
            - **先验分布参数 (`μ`, `σ`)**: 这是一个序列，每个音素对应一组 `(μ_i, σ_i)`。这两个参数定义了模型对每个音素应该如何发音的“先验知识”。
    2. **时长预测器 (Duration Predictor) 部分:**
        - 将编码器的隐藏表示 `h` 输入**时长预测器** (`f_dur`)。
        - 时长预测器输出一个预测的**时长序列 `d_hat`**。
        - **注意:** 这一步的输入 `h` 被 `stop_gradient` 包裹，这意味着从时长预测器产生的损失梯度不会影响到文本编码器。
    3. **解码器 (Decoder) 部分:**
        - 将真实的梅尔频谱图 `x` 输入**基于流的解码器** (`f_dec`)。
        - 解码器执行**逆向操作 (inverse pass)**，将 `x` 转换成一个**隐变量 `z`**。
        - 同时，解码器还会计算**雅可比行列式的对数值 (log-determinant of Jacobian)**，这是计算似然损失所必需的。
    
    至此，我们获得了计算损失所需的所有组件：
    
    - 先验分布参数 `(μ, σ)`
    - 预测时长 `d_hat`
    - 隐变量 `z`
    - 雅可比行列式对数值
    - 真实的梅尔频谱图 `x`
    
    ### 步骤 2: 独热对齐搜索 (Monotonic Alignment Search, MAS)
    
    这是Glow-TTS训练过程中的创新核心。它在**每次迭代**中动态地寻找最佳对齐。
    
    1. **输入:**
        - 解码器输出的隐变量 `z`。
        - 编码器输出的先验分布参数 `(μ, σ)`。
    2. **目标:** 找到一个 **独热（monotonic）**且 **不跳过（surjective）**的对齐路径 `A*`，使得在该对齐下，隐变量 `z` 的对数似然 `log P(z|c, A)` 最大。简单来说，就是为 `z` 的每一帧找到最匹配的那个音素 `(μ_i, σ_i)`。
        
        $$
        A := z_j \mapsto c_i
        $$
        
    3. **方法:** 使用**动态规划 (Dynamic Programming)**，类似于Viterbi算法。
        - 构建一个二维表格 `Q`，其中 `Q[i, j]` 表示“前 `j` 帧语音 (`z_1...z_j`) 与前 `i` 个音素 (`c_1...c_i`) 对齐的最大对数似然”
            
            $$
            Q[i,j] := z_1\cdots z_j \ 与\ c_1 \cdots c_i\ 对齐的最大似然 = \max_{A}\sum_{k=1}^j\log\mathcal{N}(z_k;\mu_{A(k)},\sigma_{A(k)})
            $$
            
        - 通过递推公式
            
            $$
            Q[i, j] = \max(Q[i-1, j-1], Q[i, j-1]) + \log \mathcal{N}(z_j; μ_i, σ_i) 
            $$
            
            填充整个表格。
            
            动态规划，目前是否用上了音素 $c_{i}$ (即 $z_{j-1}$ 是否属于 $c_i$)
            
            - 没用上 $c_{i}$: $Q[i-1, j-1] + \log \mathcal{N}(z_j; μ_i, σ_i)$
            - 用上了 $c_{i}$: $Q[i, j-1] + \log \mathcal{N}(z_j; μ_i, σ_i)$
        - 从表格的终点 `Q[T_text, T_mel]` 回溯，找到最佳对齐路径 `A*`。
    4. **输出:**
    - **最佳对齐 `A*`**: 一个序列，指明了 `z` 的每一帧对应哪个音素。
    - **"真实"时长 `d`**: 通过统计 `A*` 中每个音素出现的次数，得到每个音素的“真实”时长。
    
    ### 步骤 3: 计算损失 (Loss Calculation)
    
    现在我们有了所有必要信息，可以计算总损失。
    
    1. **最大似然损失 (`L_mle`):**
        - 使用步骤2中找到的最佳对齐 `A*`，将隐变量 `z` 与对应的 `(μ, σ)` 配对。
        - 计算 `z` 在这个对齐下的**对数似然**。
        - 加上解码器计算出的**雅可比行列式对数值**。
        - 取负数，得到 `L_mle`。这个损失会同时优化 **编码器** 和 **解码器**。
    2. **时长预测损失 (`L_dur`):**
        - 使用步骤1中预测的**时长 `d_hat`** 和步骤2中得到的“真实”时长 `d`。
        - 在对数域计算它们的**均方误差 (MSE)**，得到 `L_dur`。这个损失只优化时长预测器。
    3. **总损失 (`L_total`):**
        - 将两个损失相加: `L_total = L_mle + L_dur`。
    
    ### 步骤 4: 反向传播与参数更新 (Backward Pass & Update)
    
    1. **计算梯度:** 对总损失 `L_total` 进行反向传播，计算模型中所有参数（编码器、解码器、时长预测器）的梯度。
    2. **更新参数:** 使用优化器（如Adam）根据计算出的梯度更新模型的权重。
    
    ### 总结
    
    Glow-TTS的训练是一个巧妙的迭代过程，可以看作是**期望最大化 (Expectation-Maximization, EM)** 算法思想的一种体现：
    
    - **E-Step (期望步):** 在固定的模型参数下，通过**MAS算法**找到最可能的隐变量（对齐 `A*`）。
    - **M-Step (最大化步):** 在固定的对齐 `A*` 下，通过梯度下降更新模型参数，以最大化数据的对数似然（并最小化时长预测误差）。
    
    这个过程不断重复，模型会逐渐学会如何自己找到文本和语音之间的对齐，从而生成高质量且节奏正确的语音，完全无需任何外部预训练的对齐模型。
    

$L_{mle}$ 的值可以是负数，因为此时的似然 $P(x\mid c)$ 是概率密度，因为 mel-spectrogram 是连续函数

### Dataloader

```python
# 完整的数据流程：

# 1. 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collate_mel_text,  # 指定 collate_fn
    shuffle=True
)

# 2. 训练循环
for batch in dataloader:  # 🔥 这里触发 collate_fn
    # batch 已经是处理好的批次数据
    
    # 3. 模型处理
    formatted_batch = model.format_batch(batch)  # 🔥 这里调用 format_batch
    
    # 4. 前向传播
    outputs = model(formatted_batch)
```

1. **调用顺序**: Dataset.__getitem__ → DataLoader → collate_fn → `model.format_batch` → `model.forward`
2. **collate_fn**: 在 DataLoader 迭代时自动调用，将单个样本组合成批次
3. **format_batch**: 在模型的 train_step/eval_step 中手动调用，进一步处理批次数据

存储的数据 [B, C, T]

batch 中的数据 [B, C, T]

inference/forward 输出中的mel [B, T, C]

dataclass python装饰器

### 其他选择？

- Tacotron: [paper](https://arxiv.org/abs/1703.10135) ❌
- Tacotron2: [paper](https://arxiv.org/abs/1712.05884) ❌
- Glow-TTS: [paper](https://arxiv.org/abs/2005.11129) ✅
- Speedy-Speech: [paper](https://arxiv.org/abs/2008.03802) ❓
- Align-TTS: [paper](https://arxiv.org/abs/2003.01950) ❓
- FastPitch: [paper](https://arxiv.org/pdf/2006.06873.pdf)
    - 类似 FastSpeech
- FastSpeech: [paper](https://arxiv.org/abs/1905.09263)
- FastSpeech2: [paper](https://arxiv.org/abs/2006.04558) ❌
    - 做不了
- SC-GlowTTS: [paper](https://arxiv.org/abs/2104.05557) ❌
    - GlowTTS 的 multi-speaker 版本
- Capacitron: [paper](https://arxiv.org/abs/1906.03402)
- OverFlow: [paper](https://arxiv.org/abs/2211.06892) ❓
- Neural HMM TTS: [paper](https://arxiv.org/abs/2108.13320) ❌
    - OverFlow 的前作
- Delightful TTS: [paper](https://arxiv.org/abs/2110.12612)
    - 很像 fastspeech2
- StableTTS ❌
    - 太慢
- MELLE ❌
    - 没开源
- StyleTTS2 ❓

![image.png](GlowTTS%2022cb1baf8a678027a798c293618a8f5a/image%202.png)

> from “OverFlow”
> 

### 尝试拟合一条数据

训练用的 mel-spectrogram

![output1.png](GlowTTS%2022cb1baf8a678027a798c293618a8f5a/output1.png)

模型生成的 mel-spectrogram

![output.png](GlowTTS%2022cb1baf8a678027a798c293618a8f5a/output.png)

由训练数据通过 BigVGAN 生成的音频

[output1.wav](GlowTTS%2022cb1baf8a678027a798c293618a8f5a/output1.wav)

由生成的频谱图通过 BigVGAN 生成的音频

[output.wav](GlowTTS%2022cb1baf8a678027a798c293618a8f5a/output.wav)

基本能听出声音，但是电流声很大