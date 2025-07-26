#!/bin/bash

# 定义路径
wav_root_path="./data/AItts_wav"
save_root_path="./data/AItts"

# 检查 wav_root_path 是否存在
if [ ! -d "$wav_root_path" ]; then
    echo "错误：目录 $wav_root_path 不存在"
    exit 1
fi

# 创建 save_root_path 目录
mkdir -p "$save_root_path" || { echo "无法创建目录 $save_root_path"; exit 1; }

# 循环处理子目录
for idx in 1 2 3 4 5 "yue"; do
    wav_dir="$wav_root_path/$idx"
    save_dir="$save_root_path/$idx"
    
    # 检查子目录是否存在
    if [ ! -d "$wav_dir" ]; then
        echo "警告：子目录 $wav_dir 不存在，跳过处理"
        continue
    fi
    
    echo "处理 $wav_dir 到 $save_dir ..."
    python Wav_to_Mel/parallel_wav_to_melspec.py --wave_path "$wav_dir" --save_path "$save_dir" || {
        echo "处理 $wav_dir 失败"
        exit 1
    }
done

echo "所有处理完成"    