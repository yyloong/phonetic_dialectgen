# 讯飞语音识别数据结构说明

## 📁 保存的文件说明

### 1. 基础结果文件
- `spark_iat_results_YYYYMMDD_HHMMSS.json` - 处理后的基础结果
- `spark_iat_progress.json` - 进度跟踪文件

### 2. 原始数据文件 （⭐ 重要！用于AI训练）
- `spark_iat_raw_results_YYYYMMDD_HHMMSS.json` - 完整的原始响应数据
- `spark_iat_progress_raw.json` - 实时原始数据跟踪

## 🔍 详细数据结构

### 原始响应数据结构 (spark_iat_progress_raw.json)

```json
{
  "raw_responses": {
    "shorts_2025-07-01_10-48-16": {
      "files": ["2025-07-01_10-48-16_PART0000.wav", "2025-07-01_10-48-16_PART0001.wav"],
      "timestamp": "2025-01-15T10:30:05.123456",
      "detailed_results": [
        {
          "timestamp": "2025-01-15T10:30:05.123456",
          "decoded_text": {
            "bg": 0,          // 开始时间戳 (ms)
            "ed": 2280,       // 结束时间戳 (ms)
            "ls": false,      // 是否最后一块结果
            "sn": 1,          // 结果序号
            "pgs": "rpl",     // 流式识别操作方式 (rpl=替换, apd=追加)
            "rst": "rlt",     // 结果类型 (rlt=最终结果, pgs=流式结果)
            "rg": [1, 14],    // 结果标识范围
            "ws": [           // 词结构数组 (重要！)
              {
                "wb": 0,      // 词边界
                "wc": 0.85,   // 词置信度
                "we": 480,    // 词结束时间
                "wp": "n",    // 词性 (n=名词, p=标点)
                "cw": [       // 候选词数组
                  {
                    "sc": 0.95,  // 候选词置信度
                    "w": "科大讯飞"  // 候选词内容
                  },
                  {
                    "sc": 0.85,
                    "w": "科大迅飞"  // 第二候选
                  }
                ]
              }
            ]
          },
          "bg": 0,
          "ed": 2280,
          "ls": false,
          "sn": 1,
          "pgs": "rpl",
          "rst": "rlt",
          "rg": [1, 14],
          "ws": [...],      // 完整词结构
          "group_key": "shorts_2025-07-01_10-48-16"
        }
      ],
      "raw_messages": [
        {
          "timestamp": "2025-01-15T10:30:05.123456",
          "raw_message": {
            "header": {
              "code": 0,
              "message": "success",
              "sid": "ase000e065f@dx193f81b97fb750c882",
              "status": 2
            },
            "payload": {
              "result": {
                "encoding": "utf8",
                "compress": "raw",
                "format": "json",
                "text": "eyJzbiI6MSwibHMiO..."  // base64编码的响应
              }
            }
          },
          "group_key": "shorts_2025-07-01_10-48-16"
        }
      ]
    }
  },
  "last_update": "2025-01-15T10:30:05.123456"
}
```

## 🎯 用于AI训练的关键数据

### 1. 时间对齐数据
- `bg` (begin): 语音开始时间戳 (毫秒)
- `ed` (end): 语音结束时间戳 (毫秒)
- `wb` (word begin): 词级开始时间
- `we` (word end): 词级结束时间

### 2. 多候选数据
- `cw` (candidate words): 候选词数组
- `sc` (score): 每个候选词的置信度
- `nbest=5`: 句子级多候选
- `wbest=5`: 词级多候选

### 3. 语言学特征
- `wp` (word part): 词性标注
  - `"n"`: 名词
  - `"p"`: 标点符号
  - `"v"`: 动词
  - 等等...
- `w` (word): 词内容（包含标点符号）

### 4. 流式识别信息
- `pgs` (process): 处理状态
  - `"rpl"`: 替换前一次结果
  - `"apd"`: 追加到前一次结果
- `rst` (result): 结果类型
  - `"rlt"`: 最终结果
  - `"pgs"`: 流式中间结果

## 🔧 使用示例

### Python代码示例：提取训练数据

```python
import json
from pathlib import Path

def extract_training_data(raw_file_path):
    """从原始数据中提取训练用数据"""
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_data = []
    
    for group_key, group_data in data['raw_responses'].items():
        for result in group_data.get('detailed_results', []):
            # 提取时间对齐信息
            segment = {
                'audio_group': group_key,
                'start_time': result['bg'],
                'end_time': result['ed'],
                'words': []
            }
            
            # 提取词级信息
            for word_info in result.get('ws', []):
                word_data = {
                    'start_time': word_info.get('wb', 0),
                    'end_time': word_info.get('we', 0),
                    'word_type': word_info.get('wp', ''),
                    'candidates': []
                }
                
                # 提取候选词
                for candidate in word_info.get('cw', []):
                    word_data['candidates'].append({
                        'text': candidate.get('w', ''),
                        'confidence': candidate.get('sc', 0.0)
                    })
                
                segment['words'].append(word_data)
            
            training_data.append(segment)
    
    return training_data

# 使用示例
training_data = extract_training_data('spark_iat_progress_raw.json')
```

## 📊 数据统计和分析

### 检查数据完整性

```python
def analyze_data_completeness(raw_file_path):
    """分析数据完整性"""
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_groups = len(data['raw_responses'])
    total_segments = 0
    total_words = 0
    total_candidates = 0
    
    for group_key, group_data in data['raw_responses'].items():
        segments = len(group_data.get('detailed_results', []))
        total_segments += segments
        
        for result in group_data.get('detailed_results', []):
            words = len(result.get('ws', []))
            total_words += words
            
            for word_info in result.get('ws', []):
                candidates = len(word_info.get('cw', []))
                total_candidates += candidates
    
    print(f"音频组数: {total_groups}")
    print(f"语音段数: {total_segments}")
    print(f"词数: {total_words}")
    print(f"候选词数: {total_candidates}")
    print(f"平均每个词的候选数: {total_candidates/total_words:.2f}")
```

## 🎓 训练建议

### 1. 语音识别模型训练
- 使用 `bg`/`ed` 时间戳对齐音频和文本
- 利用多候选数据进行 N-best 训练
- 使用置信度信息进行样本加权

### 2. 语言模型训练
- 提取标点符号信息训练标点预测模型
- 使用词性信息进行语法分析
- 利用多候选信息进行语言模型校准

### 3. 说话人适应
- 使用时间戳信息进行说话人分段
- 基于置信度筛选高质量样本
- 利用流式结果进行在线适应

## ⚠️ 注意事项

1. **数据完整性**：每次识别后立即保存，防止数据丢失
2. **文件大小**：原始数据文件可能很大，注意存储空间
3. **版本控制**：不要将包含敏感信息的原始数据提交到版本控制
4. **备份策略**：定期备份原始数据文件
5. **隐私保护**：确保语音内容符合隐私保护要求

## 📞 技术支持

如果在使用过程中遇到问题：
1. 检查日志文件中的详细错误信息
2. 验证原始数据文件的完整性
3. 确认API配置和网络连接正常
4. 查看 `spark_iat_progress.json` 了解处理进度 