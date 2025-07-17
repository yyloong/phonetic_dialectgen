import requests
import csv
import re
import time

def ask_ollama(prompt, model="gemma3:27b"):
    """调用Ollama API"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 1.2,
    }
    
    try:
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"API调用错误: {e}")
        return ""

def extract_sentences(text):
    """从返回的文本中提取句子"""
    sentences = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 匹配格式：数字. 句子内容
        match = re.match(r'^\d+\.\s*(.+)$', line)
        if match:
            sentence = match.group(1).strip()
            sentences.append(sentence)
    
    return sentences

def generate_sentences_batch():
    """生成一批句子（20个）"""
    prompt = """请随机生成20个句子，所有句子需要包含说话的话语，主题和内容可以天马行空, 句式尽量多样化。每个句子长度在10到20个字之间。输出格式为一行一个句子，前面带有序号，格式为：1. 句子内容
2. 句子内容
..."""
    
    response = ask_ollama(prompt)
    if response:
        return extract_sentences(response)
    return []

def main():
    print("开始生成1000个随机句子...")
    
    all_sentences = []
    target_count = 1000
    batch_size = 20
    
    file = 'conv.csv'
    # 创建CSV文件并写入表头
    with open(file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['序号', '句子'])
        
        current_number = 1
        
        while len(all_sentences) < target_count:
            print(f"正在生成第 {len(all_sentences)+1}-{min(len(all_sentences)+batch_size, target_count)} 个句子...")
            
            # 生成一批句子
            batch_sentences = generate_sentences_batch()
            
            if not batch_sentences:
                print("生成失败，重试中...")
                time.sleep(2)
                continue
            
            # 写入CSV文件
            for sentence in batch_sentences:
                if len(all_sentences) >= target_count:
                    break
                    
                writer.writerow([current_number, sentence])
                all_sentences.append(sentence)
                current_number += 1
            
            # 避免请求过于频繁
            time.sleep(0.2)
    
    print(f"\n完成！共生成 {len(all_sentences)} 个句子")
    print("文件保存路径: ", file)
    
    # 显示前几个句子作为示例
    print("\n前5个句子示例:")
    for i, sentence in enumerate(all_sentences[:5], 1):
        print(f"{i}. {sentence}")

if __name__ == "__main__":
    main()