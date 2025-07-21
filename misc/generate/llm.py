import requests
import json

def ask_gemma3(question, model="gemma3"):
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": question,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "未获取到回答")
        
    except requests.exceptions.ConnectionError:
        return "错误：无法连接到Ollama服务。请确保Ollama正在运行。"
    except requests.exceptions.Timeout:
        return "错误：请求超时。"
    except requests.exceptions.RequestException as e:
        return f"错误：请求失败 - {str(e)}"
    except json.JSONDecodeError:
        return "错误：响应格式错误。"

def main():
    print("=== Ollama Gemma3 问答程序 ===")
    print("输入 'quit' 或 'exit' 退出程序")
    print()
    
    while True:
        question = input("请输入你的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not question:
            print("请输入一个有效的问题。")
            continue
        
        print("\n正在思考中...")
        answer = ask_gemma3(question)
        print(f"\n回答: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()