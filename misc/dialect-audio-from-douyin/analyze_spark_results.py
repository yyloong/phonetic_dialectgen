import json
import csv
import os
from pathlib import Path
import datetime

def analyze_json_structure(json_file):
    """分析JSON文件的结构"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"JSON文件包含 {len(data)} 个音频组")
        print(f"前5个键名: {list(data.keys())[:5]}")
        
        # 查看第一个条目的结构
        first_key = list(data.keys())[0]
        first_entry = data[first_key]
        print(f"\n第一个条目的结构:")
        print(f"键名: {first_key}")
        print(f"包含的字段: {first_entry.keys()}")
        print(f"文件数量: {len(first_entry.get('files', []))}")
        
        if 'detailed_results' in first_entry:
            print(f"详细结果数量: {len(first_entry['detailed_results'])}")
            if first_entry['detailed_results']:
                first_detail = first_entry['detailed_results'][0]
                print(f"第一个详细结果的字段: {first_detail.keys()}")
                
                # 查看decoded_text结构
                if 'decoded_text' in first_detail:
                    decoded = first_detail['decoded_text']
                    print(f"decoded_text的字段: {decoded.keys()}")
                    
                    # 查看vad结构
                    if 'vad' in decoded:
                        vad = decoded['vad']
                        print(f"vad字段: {vad}")
                        if 'ws' in vad and vad['ws']:
                            vad_ws = vad['ws'][0]
                            print(f"vad.ws第一个元素: {vad_ws}")
                    
                    # 查看ws结构
                    if 'ws' in decoded and decoded['ws']:
                        ws_first = decoded['ws'][0]
                        print(f"ws第一个元素的字段: {ws_first.keys()}")
                        
                        if 'cw' in ws_first and ws_first['cw']:
                            cw_first = ws_first['cw'][0]
                            print(f"cw第一个元素的字段: {cw_first.keys()}")
                            print(f"cw第一个元素示例: {cw_first}")
        
        return data
        
    except Exception as e:
        print(f"分析JSON文件时出错: {e}")
        return None

def extract_candidates_to_csv(data, output_csv):
    """提取候选词/句子/标点符号到CSV文件"""
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'group_key', 'audio_directory', 'audio_files', 'balanced_file_path',
            'text_content', 'confidence_score', 'word_type',
            'vad_start_time_ms', 'vad_end_time_ms', 'vad_duration_ms',
            'word_start_time_ms', 'word_end_time_ms', 'word_duration_ms',
            'sentence_number', 'is_last_result', 'pgs_type', 'rst_type',
            'phone_info', 'language_info'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for group_key, group_data in data.items():
            # 解析group_key获取目录信息
            if group_key.startswith('shorts_noisy_'):
                audio_directory = 'shorts_noisy'
                timestamp = group_key.replace('shorts_noisy_', '')
            elif group_key.startswith('shorts_'):
                audio_directory = 'shorts'
                timestamp = group_key.replace('shorts_', '')
            elif group_key.startswith('balanced_'):
                audio_directory = 'balanced'
                timestamp = group_key.replace('balanced_', '')
            else:
                # 从group_key推断
                parts = group_key.split('_')
                if len(parts) >= 2:
                    audio_directory = parts[0]
                    timestamp = '_'.join(parts[1:])
                else:
                    audio_directory = 'unknown'
                    timestamp = group_key
            
            # 构建balanced目录下的对应文件路径
            balanced_file_path = f"balanced/{timestamp}.wav"
            
            audio_files = ', '.join(group_data.get('files', []))
            
            # 处理详细结果
            detailed_results = group_data.get('detailed_results', [])
            
            for result in detailed_results:
                decoded_text = result.get('decoded_text', {})
                
                # 从vad字段提取时间信息
                vad_start_time = 0
                vad_end_time = 0
                vad_duration = 0
                
                vad_info = decoded_text.get('vad', {})
                if 'ws' in vad_info and vad_info['ws']:
                    vad_ws = vad_info['ws'][0]  # 取第一个VAD段
                    vad_start_time = vad_ws.get('bg', 0)
                    vad_end_time = vad_ws.get('ed', 0)
                    vad_duration = vad_end_time - vad_start_time if vad_end_time > vad_start_time else 0
                
                # 提取其他信息
                sentence_number = decoded_text.get('sn', 0)
                is_last_result = decoded_text.get('ls', False)
                pgs_type = decoded_text.get('pgs', '')
                rst_type = decoded_text.get('rst', '')
                
                # 提取词信息
                ws_list = decoded_text.get('ws', [])
                
                for ws_item in ws_list:
                    word_bg = ws_item.get('bg', 0)  # 词在整个音频中的开始位置
                    
                    # 提取候选词
                    cw_list = ws_item.get('cw', [])
                    
                    for cw_item in cw_list:
                        text_content = cw_item.get('w', '')
                        confidence_score = cw_item.get('sc', 0)
                        word_type = cw_item.get('wp', '')  # 词性信息
                        
                        # 词级时间信息
                        word_start_time = cw_item.get('wb', 0)
                        word_end_time = cw_item.get('we', 0)
                        word_duration = word_end_time - word_start_time if word_end_time > word_start_time else 0
                        
                        # 其他信息
                        phone_info = cw_item.get('ph', '')
                        language_info = cw_item.get('lg', '')
                        
                        # 判断文本类型
                        if word_type == 'p':
                            text_type = 'punctuation'
                        elif word_type == 'n':
                            text_type = 'word'
                        else:
                            text_type = 'unknown'
                        
                        # 写入CSV行
                        writer.writerow({
                            'group_key': group_key,
                            'audio_directory': audio_directory,
                            'audio_files': audio_files,
                            'balanced_file_path': balanced_file_path,
                            'text_content': text_content,
                            'confidence_score': confidence_score,
                            'word_type': text_type,
                            'vad_start_time_ms': vad_start_time,
                            'vad_end_time_ms': vad_end_time,
                            'vad_duration_ms': vad_duration,
                            'word_start_time_ms': word_start_time,
                            'word_end_time_ms': word_end_time,
                            'word_duration_ms': word_duration,
                            'sentence_number': sentence_number,
                            'is_last_result': is_last_result,
                            'pgs_type': pgs_type,
                            'rst_type': rst_type,
                            'phone_info': phone_info,
                            'language_info': language_info
                        })

def generate_markdown_documentation():
    """生成Markdown文档说明JSON结构"""
    markdown_content = """# 讯飞语音识别结果JSON文件结构说明

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
"""

    with open('spark_iat_json_structure.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print("Markdown文档已生成: spark_iat_json_structure.md")

def main():
    # 处理完整的JSON文件
    json_file = 'spark_iat_raw_results_20250716_134734.json'
    csv_file = 'spark_iat_analysis_results_full.csv'
    
    print("=== 讯飞语音识别结果分析 (完整版) ===\n")
    
    # 1. 分析JSON结构
    print("1. 分析JSON文件结构...")
    data = analyze_json_structure(json_file)
    
    if data is None:
        print("无法读取JSON文件，程序终止。")
        return
    
    # 2. 生成Markdown文档
    print("\n2. 生成Markdown结构文档...")
    generate_markdown_documentation()
    
    # 3. 提取数据到CSV
    print("\n3. 提取数据到CSV文件...")
    extract_candidates_to_csv(data, csv_file)
    print(f"CSV文件已生成: {csv_file}")
    
    # 4. 统计信息
    print("\n4. 统计信息:")
    total_groups = len(data)
    total_results = sum(len(group.get('detailed_results', [])) for group in data.values())
    
    # 统计词汇数量和时间信息
    total_words = 0
    total_punctuation = 0
    valid_vad_segments = 0
    
    for group_data in data.values():
        for result in group_data.get('detailed_results', []):
            decoded = result.get('decoded_text', {})
            
            # 检查VAD信息
            vad_info = decoded.get('vad', {})
            if 'ws' in vad_info and vad_info['ws']:
                valid_vad_segments += 1
            
            for ws in decoded.get('ws', []):
                for cw in ws.get('cw', []):
                    if cw.get('wp') == 'p':
                        total_punctuation += 1
                    else:
                        total_words += 1
    
    print(f"- 音频组数量: {total_groups}")
    print(f"- 识别结果数量: {total_results}")
    print(f"- 有效VAD段数量: {valid_vad_segments}")
    print(f"- 词汇数量: {total_words}")
    print(f"- 标点符号数量: {total_punctuation}")
    print(f"- 总计文本单元: {total_words + total_punctuation}")
    
    # 5. 目录分布统计
    print("\n5. 目录分布统计:")
    directory_stats = {}
    for group_key in data.keys():
        if group_key.startswith('shorts_noisy_'):
            directory = 'shorts_noisy'
        elif group_key.startswith('shorts_'):
            directory = 'shorts'
        elif group_key.startswith('balanced_'):
            directory = 'balanced'
        else:
            directory = 'unknown'
        
        directory_stats[directory] = directory_stats.get(directory, 0) + 1
    
    for directory, count in directory_stats.items():
        print(f"- {directory}: {count} 个音频组")
    
    # 6. 时间信息统计
    print("\n6. 时间信息统计:")
    vad_times = []
    word_times = []
    
    for group_data in data.values():
        for result in group_data.get('detailed_results', []):
            decoded = result.get('decoded_text', {})
            
            # 收集VAD时间信息
            vad_info = decoded.get('vad', {})
            if 'ws' in vad_info and vad_info['ws']:
                vad_segment = vad_info['ws'][0]
                vad_start = vad_segment.get('bg', 0)
                vad_end = vad_segment.get('ed', 0)
                if vad_end > vad_start:
                    vad_times.append(vad_end - vad_start)
            
            # 收集词时间信息
            for ws in decoded.get('ws', []):
                for cw in ws.get('cw', []):
                    word_start = cw.get('wb', 0)
                    word_end = cw.get('we', 0)
                    if word_end > word_start:
                        word_times.append(word_end - word_start)
    
    if vad_times:
        avg_vad_duration = sum(vad_times) / len(vad_times)
        print(f"- 平均VAD段长度: {avg_vad_duration:.2f}ms")
        print(f"- VAD段长度范围: {min(vad_times):.2f}ms - {max(vad_times):.2f}ms")
    
    if word_times:
        avg_word_duration = sum(word_times) / len(word_times)
        print(f"- 平均词长度: {avg_word_duration:.2f}ms")
        print(f"- 词长度范围: {min(word_times):.2f}ms - {max(word_times):.2f}ms")
    
    # 7. 示例数据展示
    print("\n7. 示例数据展示:")
    example_count = 0
    for group_key, group_data in data.items():
        if example_count >= 3:  # 只展示前3个组的示例
            break
        
        print(f"\n示例组: {group_key}")
        print(f"对应的完整音频文件: balanced/{group_key.split('_', 1)[1]}.wav")
        print(f"包含的分片文件: {len(group_data.get('files', []))} 个")
        
        for i, result in enumerate(group_data.get('detailed_results', [])[:2]):  # 只显示前2个结果
            decoded = result.get('decoded_text', {})
            vad_info = decoded.get('vad', {})
            
            if 'ws' in vad_info and vad_info['ws']:
                vad_segment = vad_info['ws'][0]
                vad_start = vad_segment.get('bg', 0)
                vad_end = vad_segment.get('ed', 0)
                
                print(f"  结果 {i+1}: VAD时间段 {vad_start}ms - {vad_end}ms")
                
                # 显示前几个词
                word_count = 0
                for ws in decoded.get('ws', []):
                    for cw in ws.get('cw', []):
                        if word_count >= 3:  # 只显示前3个词
                            break
                        text = cw.get('w', '')
                        wb = cw.get('wb', 0)
                        we = cw.get('we', 0)
                        if text and wb > 0 and we > wb:
                            print(f"    词: '{text}' 时间: {wb}ms-{we}ms")
                            word_count += 1
                    if word_count >= 3:
                        break
        
        example_count += 1
    
    print(f"\n分析完成！结果文件:")
    print(f"- JSON结构文档: spark_iat_json_structure.md")
    print(f"- CSV数据文件: {csv_file}")
    print(f"\n修正内容:")
    print(f"✓ 修正时间信息提取 - 使用vad字段")
    print(f"✓ 修正文件路径映射 - 指向balanced目录")
    print(f"✓ 增加词级时间信息")
    print(f"✓ 增加拼音和语言信息")
    print(f"✓ 处理完整数据集({total_groups}个音频组)")
    
    # 8. 数据质量检查
    print(f"\n8. 数据质量检查:")
    groups_with_vad = sum(1 for group_data in data.values() 
                         for result in group_data.get('detailed_results', [])
                         if result.get('decoded_text', {}).get('vad', {}).get('ws'))
    
    groups_with_words = sum(1 for group_data in data.values() 
                           for result in group_data.get('detailed_results', [])
                           if result.get('decoded_text', {}).get('ws'))
    
    print(f"- 包含VAD信息的结果: {groups_with_vad}/{total_results} ({groups_with_vad/total_results*100:.1f}%)")
    print(f"- 包含词信息的结果: {groups_with_words}/{total_results} ({groups_with_words/total_results*100:.1f}%)")
    print(f"- 数据完整性: 良好" if groups_with_vad > total_results * 0.9 else "- 数据完整性: 需要检查")

if __name__ == "__main__":
    main() 