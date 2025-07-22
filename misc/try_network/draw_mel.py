import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_mel_spectrogram(pt_file_path, output_path=None):
    """
    从 .pt 文件加载梅尔频谱并绘制热图。
    
    参数:
        pt_file_path (str): .pt 文件路径
        output_path (str, optional): 保存图像的路径，若为 None 则显示图像
    """
    try:
        # 加载 .pt 文件
        mel_spectrogram = torch.load(pt_file_path)
        
        # 检查张量形状并转换为 numpy
        if isinstance(mel_spectrogram, torch.Tensor):
            # 处理可能的形状: (mel_len, mel_channels) 或 (batch_size, mel_len, mel_channels)
            if len(mel_spectrogram.shape) == 3:
                # 假设 batch_size=1，取第一个样本
                if mel_spectrogram.shape[0] != 1:
                    raise ValueError(f"Expected batch_size=1, got shape {mel_spectrogram.shape}")
                mel_spectrogram = mel_spectrogram[0]  # (mel_len, mel_channels)
            elif len(mel_spectrogram.shape) != 2:
                raise ValueError(f"Expected 2D or 3D tensor, got shape {mel_spectrogram.shape}")
            
            # 转换为 numpy 数组，形状 (mel_len, mel_channels)
            mel_np = mel_spectrogram.cpu().numpy()
        else:
            raise ValueError("Loaded data is not a PyTorch tensor")

        # 创建热图
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_np.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude (dB)')
        plt.xlabel('Time Steps')
        plt.ylabel('Mel Frequency Bins')
        plt.title('Mel Spectrogram')

        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Mel spectrogram saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

    except Exception as e:
        print(f"Error: {str(e)}")

# 示例使用
if __name__ == "__main__":
    # 替换为您的 .pt 文件路径
    pt_file_path = "../melspec/9989.pt"
    
    # 可选：指定输出图像路径
    output_path = "9989.png"
    
    # 调用可视化函数
    visualize_mel_spectrogram(pt_file_path, output_path)
    pt_file_path = "try.pt"
    output_path = "try.png"
    visualize_mel_spectrogram(pt_file_path, output_path)