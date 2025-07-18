import os
from pathlib import Path
from pydub import AudioSegment
import datetime
import numpy as np
from tqdm import tqdm

# 全局日志文件路径
LOG_FILE = None

def print_and_write_log(message):
    """统一的打印和写入日志接口"""
    print(message)
    if LOG_FILE:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def calculate_rms(audio_segment):
    """计算音频的RMS值"""
    try:
        audio_data = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            audio_data = audio_data.reshape((-1, 2))
            audio_data = audio_data.mean(axis=1)
        
        # 检查数据有效性
        if len(audio_data) == 0:
            return 0.0
        
        # 计算RMS，避免数值问题
        mean_square = np.mean(audio_data.astype(np.float64)**2)
        if mean_square < 0 or not np.isfinite(mean_square):
            return 0.0
        
        rms = np.sqrt(mean_square)
        if not np.isfinite(rms):
            return 0.0
        
        return rms
    except Exception as e:
        print_and_write_log(f"计算RMS时出错: {e}")
        return 0.0


def normalize_audio_rms(audio_segment, target_rms=None):
    """
    基于RMS值归一化音频
    
    Args:
        audio_segment: 输入音频段
        target_rms: 目标RMS值，如果为None则使用默认值
    
    Returns:
        归一化后的音频段
    """
    if target_rms is None:
        # 设置一个合理的目标RMS值（相对于16位音频的最大值）
        target_rms = 0.2 * (2**15)  # 大约是最大音量的20%
    
    current_rms = calculate_rms(audio_segment)
    
    # 检查RMS值的有效性
    if current_rms == 0 or not np.isfinite(current_rms) or not np.isfinite(target_rms) or target_rms <= 0:
        return audio_segment
    
    # 计算增益
    try:
        gain_db = 20 * np.log10(target_rms / current_rms)
        
        # 检查增益值的有效性
        if not np.isfinite(gain_db):
            return audio_segment
        
        # 限制增益范围，避免过度放大或缩小
        gain_db = max(-30, min(30, gain_db))
        
        # 应用增益
        normalized_audio = audio_segment + gain_db
        
        return normalized_audio
    except Exception as e:
        print_and_write_log(f"归一化时出错: {e}")
        return audio_segment


def normalize_audio_peak(audio_segment, target_peak_db=-3.0):
    """
    基于峰值归一化音频

    Args:
        audio_segment: 输入音频段
        target_peak_db: 目标峰值(dB)，默认-3dB

    Returns:
        归一化后的音频段
    """
    # 获取当前峰值
    current_peak_db = audio_segment.max_dBFS

    if current_peak_db == float("-inf"):
        return audio_segment

    # 计算需要的增益
    gain_db = target_peak_db - current_peak_db

    # 限制增益范围
    gain_db = max(-30, min(30, gain_db))

    # 应用增益
    normalized_audio = audio_segment + gain_db

    return normalized_audio


def analyze_audio_levels(wavs_dir):
    """分析所有音频文件的音量水平"""
    wav_files = list(wavs_dir.glob("*.wav"))

    if not wav_files:
        print_and_write_log("没有找到音频文件")
        return None

    print_and_write_log(f"分析 {len(wav_files)} 个音频文件的音量水平...")

    rms_values = []
    peak_values = []
    file_info = []

    for wav_file in tqdm(wav_files, desc="分析音频", ncols=100):
        try:
            audio = AudioSegment.from_wav(wav_file)
            rms = calculate_rms(audio)
            peak_db = audio.max_dBFS

            # 只有当RMS和峰值都有效时才添加到统计中
            if np.isfinite(rms) and rms > 0 and np.isfinite(peak_db) and peak_db > float('-inf'):
                rms_values.append(rms)
                peak_values.append(peak_db)
                file_info.append(
                    {
                        "file": wav_file.name,
                        "rms": rms,
                        "peak_db": peak_db,
                        "duration": len(audio) / 1000,
                    }
                )
            else:
                print_and_write_log(f"跳过无效音频文件 {wav_file.name}: RMS={rms}, Peak={peak_db}dB")

        except Exception as e:
            print_and_write_log(f"分析文件 {wav_file} 时出错: {repr(e)}")
            continue

    if not rms_values:
        return None

    # 计算统计信息
    stats = {
        "rms_mean": np.mean(rms_values),
        "rms_median": np.median(rms_values),
        "rms_std": np.std(rms_values),
        "rms_var": np.var(rms_values),
        "rms_min": np.min(rms_values),
        "rms_max": np.max(rms_values),
        "peak_mean": np.mean(peak_values),
        "peak_median": np.median(peak_values),
        "peak_std": np.std(peak_values),
        "peak_var": np.var(peak_values),
        "peak_min": np.min(peak_values),
        "peak_max": np.max(peak_values),
        "file_count": len(file_info),
    }

    return stats, file_info


def normalize_all_audio(wavs_dir, output_dir, method="rms", target_rms=None, target_peak_db=-6.0):
    """
    归一化所有音频文件

    Args:
        wavs_dir: 输入目录
        output_dir: 输出目录
        method: 归一化方法 ('rms' 或 'peak')
        target_rms: 目标RMS值（用于RMS方法）
        target_peak_db: 目标峰值（用于peak方法）
    """
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    # 获取所有wav文件
    wav_files = list(wavs_dir.glob("*.wav"))

    if not wav_files:
        print_and_write_log("没有找到音频文件")
        return

    print_and_write_log(f"开始归一化 {len(wav_files)} 个音频文件...")
    print_and_write_log(f"使用方法: {method}")
    if method == "rms":
        print_and_write_log(f"目标RMS: {target_rms if target_rms else 'auto'}")
    else:
        print_and_write_log(f"目标峰值: {target_peak_db} dB")

    successful_count = 0
    failed_files = []

    for wav_file in tqdm(wav_files, desc="归一化音频", ncols=100):
        try:
            # 加载音频
            audio = AudioSegment.from_wav(wav_file)

            # 应用归一化
            if method == "rms":
                normalized_audio = normalize_audio_rms(audio, target_rms)
            else:
                normalized_audio = normalize_audio_peak(audio, target_peak_db)

            # 保存到输出目录，保持相同文件名
            output_file = output_dir / wav_file.name
            normalized_audio.export(output_file, format="wav")

            successful_count += 1

        except Exception as e:
            print(f"处理文件 {wav_file} 时出错: {e}")
            failed_files.append(wav_file.name)
            continue

    print_and_write_log(f"归一化完成！")
    print_and_write_log(f"成功处理: {successful_count}/{len(wav_files)} 个文件")
    print_and_write_log(f"成功率: {successful_count/len(wav_files)*100:.1f}%")

    if failed_files:
        print_and_write_log(f"失败文件: {len(failed_files)} 个")
        for failed_file in failed_files:
            print_and_write_log(f"  - {failed_file}")


def main():
    global LOG_FILE
    
    # 设置路径
    wavs_dir = Path("./wavs")
    balanced_dir = Path("./balanced")

    # 创建时间戳日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"normalize_audio_{timestamp}.log"
    LOG_FILE = log_file

    print_and_write_log("=" * 60)
    print_and_write_log("音频音量归一化程序")
    print_and_write_log("=" * 60)

    # 检查输入目录
    if not wavs_dir.exists():
        print_and_write_log(f"输入目录不存在: {wavs_dir}")
        return

    # 分析音频水平
    print_and_write_log("步骤1: 分析音频文件音量水平")
    analysis_result = analyze_audio_levels(wavs_dir)

    if analysis_result is None:
        print_and_write_log("无法分析音频文件")
        return

    original_stats, file_info = analysis_result

    # 显示原始音频分析结果
    print_and_write_log("原始音频音量分析结果:")
    print_and_write_log(f"  文件数量: {original_stats['file_count']}")
    print_and_write_log(f"  RMS 平均值: {original_stats['rms_mean']:.2f}")
    print_and_write_log(f"  RMS 中位数: {original_stats['rms_median']:.2f}")
    print_and_write_log(f"  RMS 标准差: {original_stats['rms_std']:.2f}")
    print_and_write_log(f"  RMS 方差: {original_stats['rms_var']:.2f}")
    print_and_write_log(f"  RMS 范围: {original_stats['rms_min']:.2f} - {original_stats['rms_max']:.2f}")
    print_and_write_log(f"  峰值平均值: {original_stats['peak_mean']:.2f} dB")
    print_and_write_log(f"  峰值中位数: {original_stats['peak_median']:.2f} dB")
    print_and_write_log(f"  峰值标准差: {original_stats['peak_std']:.2f} dB")
    print_and_write_log(f"  峰值方差: {original_stats['peak_var']:.2f} dB")
    print_and_write_log(f"  峰值范围: {original_stats['peak_min']:.2f} - {original_stats['peak_max']:.2f} dB")

    # 选择归一化方法
    print_and_write_log(f"步骤2: 选择归一化方法")
    print_and_write_log("推荐使用 RMS 方法，因为它更好地保持语音的相对音量关系")

    # 使用RMS方法，目标设置为中位数
    target_rms = original_stats["rms_median"]

    print_and_write_log(f"使用 RMS 方法，目标 RMS 值: {target_rms:.2f}")

    # 执行归一化
    print_and_write_log(f"步骤3: 执行音量归一化")
    normalize_all_audio(wavs_dir, balanced_dir, method="rms", target_rms=target_rms)

    # 步骤4: 分析归一化后的音频
    print_and_write_log(f"步骤4: 分析归一化后的音频")
    normalized_analysis = analyze_audio_levels(balanced_dir)
    
    if normalized_analysis is not None:
        normalized_stats, _ = normalized_analysis
        
        print_and_write_log("归一化后音频音量分析结果:")
        print_and_write_log(f"  文件数量: {normalized_stats['file_count']}")
        print_and_write_log(f"  RMS 平均值: {normalized_stats['rms_mean']:.2f}")
        print_and_write_log(f"  RMS 中位数: {normalized_stats['rms_median']:.2f}")
        print_and_write_log(f"  RMS 标准差: {normalized_stats['rms_std']:.2f}")
        print_and_write_log(f"  RMS 方差: {normalized_stats['rms_var']:.2f}")
        print_and_write_log(f"  RMS 范围: {normalized_stats['rms_min']:.2f} - {normalized_stats['rms_max']:.2f}")
        print_and_write_log(f"  峰值平均值: {normalized_stats['peak_mean']:.2f} dB")
        print_and_write_log(f"  峰值中位数: {normalized_stats['peak_median']:.2f} dB")
        print_and_write_log(f"  峰值标准差: {normalized_stats['peak_std']:.2f} dB")
        print_and_write_log(f"  峰值方差: {normalized_stats['peak_var']:.2f} dB")
        print_and_write_log(f"  峰值范围: {normalized_stats['peak_min']:.2f} - {normalized_stats['peak_max']:.2f} dB")
        
        # 对比分析
        print_and_write_log("归一化前后对比:")
        
        # 检查数值有效性
        if (np.isfinite(original_stats['rms_mean']) and np.isfinite(normalized_stats['rms_mean']) and
            np.isfinite(original_stats['rms_std']) and np.isfinite(normalized_stats['rms_std']) and
            np.isfinite(original_stats['rms_var']) and np.isfinite(normalized_stats['rms_var'])):
            
            print_and_write_log(f"  RMS 平均值: {original_stats['rms_mean']:.2f} -> {normalized_stats['rms_mean']:.2f}")
            print_and_write_log(f"  RMS 标准差: {original_stats['rms_std']:.2f} -> {normalized_stats['rms_std']:.2f}")
            print_and_write_log(f"  RMS 方差: {original_stats['rms_var']:.2f} -> {normalized_stats['rms_var']:.2f}")
            print_and_write_log(f"  RMS 范围: {original_stats['rms_max']-original_stats['rms_min']:.2f} -> {normalized_stats['rms_max']-normalized_stats['rms_min']:.2f}")
            
            # 计算降低百分比
            if original_stats['rms_std'] > 0:
                rms_std_reduction = ((original_stats['rms_std'] - normalized_stats['rms_std']) / original_stats['rms_std']) * 100
                print_and_write_log(f"  RMS 标准差降低: {rms_std_reduction:.1f}%")
            else:
                print_and_write_log(f"  RMS 标准差降低: 无法计算")
            
            if original_stats['rms_var'] > 0:
                rms_var_reduction = ((original_stats['rms_var'] - normalized_stats['rms_var']) / original_stats['rms_var']) * 100
                print_and_write_log(f"  RMS 方差降低: {rms_var_reduction:.1f}%")
            else:
                print_and_write_log(f"  RMS 方差降低: 无法计算")
        else:
            print_and_write_log("  RMS 对比: 由于数值无效，无法进行对比分析")
            print_and_write_log("  可能原因: 部分音频文件为静音或损坏")
            
        # 峰值对比总是可以显示
        print_and_write_log(f"  峰值平均值: {original_stats['peak_mean']:.2f} -> {normalized_stats['peak_mean']:.2f} dB")
        print_and_write_log(f"  峰值标准差: {original_stats['peak_std']:.2f} -> {normalized_stats['peak_std']:.2f} dB")
    else:
        print_and_write_log("无法分析归一化后的音频文件")

    print_and_write_log(f"日志已保存到: {log_file}")
    print_and_write_log("=" * 60)
    print_and_write_log("音量归一化完成！")
    print_and_write_log("现在可以运行 split_audio.py 来分割音频文件")
    print_and_write_log("=" * 60)


if __name__ == "__main__":
    main()
