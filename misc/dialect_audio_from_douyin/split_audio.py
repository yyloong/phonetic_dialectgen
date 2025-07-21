#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence
import sys
import datetime
import statistics
import numpy as np
import json
import yaml
import tomllib

# 全局变量用于记录结果
successful_files = []
failed_files = []
segment_lengths = []

# 全局日志文件路径
LOG_FILE = None


def print_and_write_log(message):
    """统一的打印和写入日志接口"""
    print(message)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def load_config(config_path):
    """加载配置文件"""
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix == ".json":
                return json.load(f)
            elif config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                return yaml.safe_load(f)
            elif config_path.suffix == ".toml":
                return tomllib.loads(f.read())
            else:
                print_and_write_log(f"不支持的配置文件格式: {config_path.suffix}")
                return {}
    except Exception as e:
        print_and_write_log(f"加载配置文件失败 {config_path}: {e}")
        return {}


def save_failed_files(failed_files, config_path):
    """保存失败文件列表到配置文件"""
    config = {
        "failed_files": failed_files,
        "relaxed_thresholds": {
            "silence_thresh": -40,  # 更宽松的静音阈值
            "min_silence_len": 300,  # 更短的最小静音长度
            "target_length": 8000,  # 更短的目标长度
            "force_split": True,  # 强制分割
        },
        "generated_at": datetime.datetime.now().isoformat(),
    }

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix == ".json":
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix == ".toml":
                import toml

                toml.dump(config, f)
        print_and_write_log(f"失败文件配置已保存到: {config_path}")
    except Exception as e:
        print_and_write_log(f"保存配置文件失败: {e}")


def get_file_config(filename, config):
    """获取文件的特定配置"""
    if not config:
        return {}

    failed_files = config.get("failed_files", [])
    if filename in failed_files:
        return config.get("relaxed_thresholds", {})

    return {}


def split_audio_by_silence(
    input_file,
    output_dir,
    config=None,
    target_length=10000,
    max_length=60000,
    min_length=6000,
    min_silence_len=500,
    silence_thresh=-50,
    force_split=False,
):
    """
    根据静音分割音频文件

    Args:
        input_file: 输入音频文件路径
        output_dir: 输出目录
        config: 配置字典，包含文件特定的处理参数
        target_length: 目标分段长度(毫秒)，默认10秒
        max_length: 最大分段长度(毫秒)，默认60秒
        min_length: 最小分段长度(毫秒)，默认6秒
        min_silence_len: 最小静音长度(毫秒)，默认500ms
        silence_thresh: 静音阈值(dB)，默认-50dB
        force_split: 是否强制分割小于60秒的音频

    Returns:
        bool: 是否成功分割
    """

    # 应用配置参数
    if config:
        target_length = config.get("target_length", target_length)
        min_silence_len = config.get("min_silence_len", min_silence_len)
        silence_thresh = config.get("silence_thresh", silence_thresh)
        force_split = config.get("force_split", force_split)

        print_and_write_log(
            f"使用特殊配置: thresh={silence_thresh}, min_silence={min_silence_len}, force={force_split}"
        )

    # 加载音频文件
    try:
        audio = AudioSegment.from_wav(input_file)
        print_and_write_log(f"处理文件: {input_file}")
        print_and_write_log(f"音频时长: {len(audio)/1000:.2f}秒")
    except Exception as e:
        print_and_write_log(f"加载文件失败 {input_file}: {e}")
        return False

    # 如果音频本身就短于最小长度，标记为失败
    if len(audio) < min_length:
        print_and_write_log(
            f"音频太短 ({len(audio)/1000:.2f}秒 < {min_length/1000}秒)，跳过: {input_file}"
        )
        return False

    # 检测静音段
    silence_ranges = detect_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )

    # 获取文件名（不含扩展名）
    base_name = Path(input_file).stem

    # 如果音频短于最大长度的处理逻辑
    if len(audio) <= max_length:
        # 如果没有检测到静音且不强制分割，作为单个片段保存
        if not silence_ranges and not force_split:
            if len(audio) >= min_length:
                output_file = output_dir / f"{base_name}_PART0000.wav"
                audio.export(output_file, format="wav")
                print_and_write_log(f"保存: {output_file} (长度: {len(audio)/1000:.2f}秒)")
                segment_lengths.append(len(audio) / 1000)
                return True
            else:
                print_and_write_log(f"音频太短无法分割: {input_file}")
                return False

        # 如果强制分割或检测到静音，继续分割流程
        if force_split and silence_ranges:
            print_and_write_log(
                f"强制分割模式: 音频长度 {len(audio)/1000:.2f}秒，检测到 {len(silence_ranges)} 个静音段"
            )
        elif not silence_ranges:
            # 强制分割但没有静音段
            if len(audio) >= min_length:
                output_file = output_dir / f"{base_name}_PART0000.wav"
                audio.export(output_file, format="wav")
                print_and_write_log(
                    f"强制分割: 无静音段，保存完整文件: {output_file} (长度: {len(audio)/1000:.2f}秒)"
                )
                segment_lengths.append(len(audio) / 1000)
                return True
            else:
                print_and_write_log(f"音频太短无法分割: {input_file}")
                return False

    # 如果没有检测到静音但音频很长，标记为失败
    if not silence_ranges and len(audio) > max_length:
        print_and_write_log(f"音频过长且没有检测到合适的静音分割点: {input_file}")
        print_and_write_log(
            f"音频长度: {len(audio)/1000:.2f}秒, 无法在{max_length/1000}秒内找到分割点"
        )
        return False

    # 根据静音分割
    segments = []
    start_time = 0

    for silence_start, silence_end in silence_ranges:
        # 计算到当前静音的时间
        segment_length = silence_start - start_time

        # 如果当前段长度合适，进行分割
        if segment_length >= min_length:
            if segment_length <= max_length:
                # 长度在合理范围内
                segments.append((start_time, silence_start))
                start_time = silence_end
            elif segment_length >= target_length * 0.8:
                # 长度接近目标长度且不超过最大长度
                segments.append((start_time, silence_start))
                start_time = silence_end
            else:
                # 段太长，需要寻找更早的分割点
                continue
        else:
            # 段太短，继续寻找下一个静音点
            continue

    # 处理最后一段
    if start_time < len(audio):
        last_segment_length = len(audio) - start_time
        if last_segment_length >= min_length:
            if last_segment_length <= max_length:
                segments.append((start_time, len(audio)))
            else:
                # 最后一段太长，需要与前一段合并或寻找分割点
                if segments:
                    # 与前一段合并
                    prev_start, prev_end = segments[-1]
                    combined_length = len(audio) - prev_start
                    if combined_length <= max_length:
                        segments[-1] = (prev_start, len(audio))
                    else:
                        # 合并后仍然太长，标记为失败
                        print_and_write_log(f"最后一段太长且无法合并: {input_file}")
                        return False
                else:
                    # 没有前一段可以合并
                    print_and_write_log(f"最后一段太长且无法分割: {input_file}")
                    return False
        else:
            # 最后一段太短，与前一段合并
            if segments:
                prev_start, prev_end = segments[-1]
                combined_length = len(audio) - prev_start
                if combined_length <= max_length:
                    segments[-1] = (prev_start, len(audio))
                else:
                    print_and_write_log(f"最后一段太短且合并后超长: {input_file}")
                    return False
            else:
                print_and_write_log(f"只有一段且太短: {input_file}")
                return False

    # 检查分割结果
    if not segments:
        print_and_write_log(f"无法找到合适的分割点: {input_file}")
        return False

    # 验证所有段都符合要求
    for i, (start, end) in enumerate(segments):
        duration = (end - start) / 1000
        if duration < min_length / 1000 or duration > max_length / 1000:
            print_and_write_log(
                f"分割后的段长度不符合要求: {input_file}, 段{i}, 长度{duration:.2f}秒"
            )
            return False

    # 导出分割后的音频
    for i, (start, end) in enumerate(segments):
        segment = audio[start:end]
        duration = len(segment) / 1000
        output_file = output_dir / f"{base_name}_PART{i:04d}.wav"
        segment.export(output_file, format="wav")
        print_and_write_log(f"保存: {output_file} (长度: {duration:.2f}秒)")
        segment_lengths.append(duration)

    return True


def calculate_statistics(lengths):
    """计算分布统计"""
    if not lengths:
        return {}

    lengths = np.array(lengths)

    stats = {
        "count": len(lengths),
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "std": np.std(lengths),
        "var": np.var(lengths),
        "q1": np.percentile(lengths, 25),
        "q3": np.percentile(lengths, 75),
        "p12.5": np.percentile(lengths, 12.5),
        "p37.5": np.percentile(lengths, 37.5),
        "p62.5": np.percentile(lengths, 62.5),
        "p87.5": np.percentile(lengths, 87.5),
    }

    return stats


def main():
    global LOG_FILE

    # 生成日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"split_audio_{timestamp}.log"
    LOG_FILE = log_file

    # 设置路径
    input_dir = Path("./balanced")  # 从balanced目录读取归一化后的音频
    output_dir = Path("./shorts")  # 分割后的文件保存到shorts目录

    # 加载配置文件
    config_path = Path("split_audio_config.json")
    config = load_config(config_path)

    # 检查是否有失败文件重试配置
    if config and config.get("failed_files"):
        output_dir = Path("./shorts_noisy")  # 重试失败文件时使用noisy目录
        print_and_write_log("检测到失败文件重试配置，使用 shorts_noisy 目录")

    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)

    # 初始化日志
    print_and_write_log("=" * 60)
    print_and_write_log("音频分割程序开始运行")
    print_and_write_log(f"从 {input_dir} 读取归一化后的音频文件")
    print_and_write_log(f"分割后的文件保存到 {output_dir}")
    print_and_write_log(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_and_write_log("=" * 60)

    # 获取所有wav文件
    wav_files = list(input_dir.glob("*.wav"))

    if not wav_files:
        print_and_write_log("在 ./balanced 目录中没有找到音频文件")
        print_and_write_log("请先运行 normalize_audio.py 来归一化音频文件")
        return

    # 确定要处理的文件列表
    if config and config.get("failed_files"):
        # 如果有配置文件，只处理失败的文件
        failed_file_names = config.get("failed_files", [])
        files_to_process = [f for f in wav_files if f.name in failed_file_names]
        print_and_write_log(f"配置文件中指定了 {len(failed_file_names)} 个失败文件")
        print_and_write_log(f"在 balanced 目录中找到 {len(files_to_process)} 个待重试文件")

        if len(files_to_process) != len(failed_file_names):
            missing_files = set(failed_file_names) - set(f.name for f in files_to_process)
            print_and_write_log(f"警告: 以下文件在 balanced 目录中未找到: {list(missing_files)}")
    else:
        # 否则处理所有文件
        files_to_process = wav_files
        print_and_write_log(f"找到 {len(files_to_process)} 个音频文件")

    # 处理文件
    for i, wav_file in enumerate(files_to_process, start=1):
        try:
            print_and_write_log(f"进度: {i}/{len(files_to_process)}")

            # 获取文件特定配置
            file_config = get_file_config(wav_file.name, config)

            success = split_audio_by_silence(
                wav_file,
                output_dir,
                config=file_config,
                force_split=True,  # 对小于60s的音频也尝试分割
            )

            if success:
                successful_files.append(wav_file.name)
            else:
                failed_files.append(wav_file.name)
            print_and_write_log("-" * 50)
        except Exception as e:
            print_and_write_log(f"处理文件 {wav_file} 时出错: {e}")
            failed_files.append(wav_file.name)
            continue

    # 计算统计信息
    success_rate = len(successful_files) / len(files_to_process) * 100 if files_to_process else 0

    print_and_write_log("=" * 60)
    print_and_write_log("处理结果统计")
    print_and_write_log("=" * 60)
    print_and_write_log(f"总文件数: {len(files_to_process)}")
    print_and_write_log(f"成功处理: {len(successful_files)}")
    print_and_write_log(f"失败文件: {len(failed_files)}")
    print_and_write_log(f"成功率: {success_rate:.2f}%")

    if failed_files:
        print_and_write_log("失败文件列表:")
        for failed_file in failed_files:
            print_and_write_log(f"  - {failed_file}")

    # 分割后音频片段统计
    if segment_lengths:
        stats = calculate_statistics(segment_lengths)
        print_and_write_log("分割后音频片段长度分布统计:")
        print_and_write_log(f"  总片段数: {stats['count']}")
        print_and_write_log(f"  平均长度: {stats['mean']:.2f}秒")
        print_and_write_log(f"  中位数: {stats['median']:.2f}秒")
        print_and_write_log(f"  最小长度: {stats['min']:.2f}秒")
        print_and_write_log(f"  最大长度: {stats['max']:.2f}秒")
        print_and_write_log(f"  标准差: {stats['std']:.2f}秒")
        print_and_write_log(f"  方差: {stats['var']:.2f}秒²")
        print_and_write_log(f"  第一四分位数 (Q1): {stats['q1']:.2f}秒")
        print_and_write_log(f"  第三四分位数 (Q3): {stats['q3']:.2f}秒")
        print_and_write_log(f"  八分位数分布:")
        print_and_write_log(f"    P12.5: {stats['p12.5']:.2f}秒")
        print_and_write_log(f"    P25.0: {stats['q1']:.2f}秒")
        print_and_write_log(f"    P37.5: {stats['p37.5']:.2f}秒")
        print_and_write_log(f"    P50.0: {stats['median']:.2f}秒")
        print_and_write_log(f"    P62.5: {stats['p62.5']:.2f}秒")
        print_and_write_log(f"    P75.0: {stats['q3']:.2f}秒")
        print_and_write_log(f"    P87.5: {stats['p87.5']:.2f}秒")

        # 长度分布直方图
        print_and_write_log(f"\n长度分布区间统计:")
        bins = [0, 6, 8, 10, 12, 15, 20, 30, 60, float("inf")]
        bin_labels = [
            "<6s",
            "6-8s",
            "8-10s",
            "10-12s",
            "12-15s",
            "15-20s",
            "20-30s",
            "30-60s",
            ">60s",
        ]

        for i in range(len(bins) - 1):
            count = sum(1 for length in segment_lengths if bins[i] <= length < bins[i + 1])
            percentage = count / len(segment_lengths) * 100
            print_and_write_log(f"  {bin_labels[i]}: {count} 个 ({percentage:.1f}%)")

        print_and_write_log("=" * 60)
    print_and_write_log(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_and_write_log("程序运行完成")

    # 保存失败文件配置
    if failed_files:
        save_failed_files([f for f in failed_files if f], config_path)
        print_and_write_log(f"失败文件配置已保存，可用于重新处理")

    print(f"\n处理完成！详细日志已保存到: {log_file}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"总片段数: {len(segment_lengths)}")
    if failed_files:
        print(f"失败文件数: {len(failed_files)}")
        print(f"失败文件配置: {config_path}")


if __name__ == "__main__":
    main()
