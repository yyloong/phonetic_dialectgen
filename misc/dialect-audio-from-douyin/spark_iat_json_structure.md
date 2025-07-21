# 讯飞语音识别结果JSON文件结构说明

## 文件概述
`spark_iat_raw_results_*.json` 文件包含了讯飞语音识别API的原始响应数据，每个音频组的详细识别结果。

## 重要发现
1. **文件路径映射**: 虽然 `files` 字段包含 `shorts/` 或 `shorts_noisy/` 目录中的分片文件，但实际的语音识别是基于 `balanced/` 目录中的完整音频文件进行的。
2. **时间信息层次**: 有两个层次的时间信息：
   - VAD层次: 从 `decoded_text.vad.ws` 中获取语音活动检测的时间段
   - 词层次: 从 `decoded_text.ws[].cw[]` 中获取每个词的精确时间位置

## 顶层结构
```json
{
  "group_key": {
    "files": [...],
    "timestamp": "...",
    "detailed_results": [...],
    "raw_messages": [...]
  }
}
```

### 字段说明

#### 1. `group_key` (字符串)
- 格式: `{目录名}_{时间戳}`
- 示例: `shorts_2022-10-28_00-00-00`
- 说明: 标识一组相关的音频文件

#### 2. `files` (数组)
- 说明: 包含该组所有音频分片文件的文件名 (位于 shorts/ 或 shorts_noisy/ 目录)
- 示例: `["2022-10-28_00-00-00_PART0000.wav", "2022-10-28_00-00-00_PART0001.wav", ...]`
- **注意**: 实际识别使用的是 balanced/ 目录下的完整音频文件

#### 3. `timestamp` (字符串)
- 格式: ISO 8601 格式
- 说明: 处理时间戳

#### 4. `detailed_results` (数组)
详细的识别结果数组，每个元素包含：

##### 4.1 `decoded_text` (对象)
识别结果的核心数据结构：

- **`sn`** (数字): 句子序号
- **`ls`** (布尔值): 是否为最后一个结果块
- **`bg`** (数字): 通常为0，不是实际的时间信息
- **`ed`** (数字): 通常为0，不是实际的时间信息
- **`vad`** (对象): **语音活动检测信息 - 包含真正的时间信息**
- **`pgs`** (字符串): 流式识别操作方式
- **`rst`** (字符串): 识别结果类型
- **`rg`** (数组): 结果标识字段
- **`ws`** (数组): 词信息数组

##### 4.2 `vad` 对象结构 ⭐ **关键时间信息**
```json
"vad": {
  "ws": [
    {
      "bg": 47,      // 语音段开始时间(毫秒)
      "ed": 1253,    // 语音段结束时间(毫秒)
      "eg": 69.21    // 能量或置信度参数
    }
  ]
}
```

##### 4.3 `ws` 数组结构
每个词槽的详细信息：

- **`bg`** (数字): 词在整个音频中的开始位置
- **`cw`** (数组): 候选词数组

##### 4.4 `cw` 数组结构
每个候选词的详细信息：

- **`w`** (字符串): 词的文本内容
- **`sc`** (数字): 置信度分数
- **`wp`** (字符串): 词性标注 ("n": 普通词, "p": 标点符号)
- **`wb`** (数字): 词开始时间(毫秒)
- **`we`** (数字): 词结束时间(毫秒)
- **`ph`** (字符串): 拼音信息
- **`lg`** (字符串): 语言信息 (如 "mandarin")
- **`wc`** (数字): 词置信度
- **`ng`** (字符串): 数字相关信息

## 时间信息说明 ⭐

### 时间层次
1. **VAD层次时间**: `decoded_text.vad.ws[].bg` 和 `decoded_text.vad.ws[].ed`
   - 表示语音活动检测到的语音段在音频中的时间范围
   - 这是句子级别的时间信息

2. **词层次时间**: `decoded_text.ws[].cw[].wb` 和 `decoded_text.ws[].cw[].we`
   - 表示每个词在音频中的精确时间位置
   - 这是词级别的时间信息

### 时间计算示例
```python
# 获取VAD时间段
vad_start = decoded_text['vad']['ws'][0]['bg']  # 如: 47ms
vad_end = decoded_text['vad']['ws'][0]['ed']    # 如: 1253ms

# 获取词的时间
word_start = cw_item['wb']  # 如: 68ms
word_end = cw_item['we']    # 如: 1155ms
```

## 文件路径映射 ⭐

### 实际文件位置
- **分片文件**: `shorts/` 或 `shorts_noisy/` 目录中的 PART*.wav 文件
- **完整音频**: `balanced/` 目录中的对应文件
- **识别基础**: 讯飞API基于 `balanced/` 目录中的完整音频文件进行识别

### 路径构建
```python
# 从group_key构建balanced路径
if group_key.startswith('shorts_'):
    timestamp = group_key.replace('shorts_', '')
    balanced_path = f"balanced/{timestamp}.wav"
```

## 数据使用示例

### 提取所有识别文本和时间
```python
for group_key, group_data in data.items():
    for result in group_data['detailed_results']:
        decoded = result['decoded_text']
        
        # 获取VAD时间段
        vad_info = decoded.get('vad', {})
        if 'ws' in vad_info and vad_info['ws']:
            vad_start = vad_info['ws'][0]['bg']
            vad_end = vad_info['ws'][0]['ed']
        
        # 获取词级信息
        for ws in decoded['ws']:
            for cw in ws['cw']:
                text = cw['w']
                word_start = cw['wb']
                word_end = cw['we']
                confidence = cw['sc']
```

## 注意事项

1. **时间信息源**: 使用 `vad` 字段中的时间信息，而不是 `decoded_text.bg/ed`
2. **文件映射**: 识别结果对应 `balanced/` 目录中的完整音频文件
3. **多候选结果**: 每个词位置有多个候选词(nbest=5, wbest=5)
4. **时间精度**: 时间戳精确到毫秒
5. **编码格式**: 所有文本内容使用UTF-8编码
