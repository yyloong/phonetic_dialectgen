#!/bin/bash
#指定载语音数据还是梅尔频谱数据
download_wav="False"

# 检查 huggingface-cli 是否安装
if ! command -v huggingface-cli &> /dev/null; then
    echo "错误：huggingface-cli 未安装，请先安装 huggingface_hub"
    echo "安装命令：pip install huggingface_hub"
    exit 1
fi

# 编译Vits/monotonic_align/core.pyx
#rm Vits/monotonic_align/*.so
#python Vits/monotonic_align/setup.py build_ext --inplace

# 定义下载目录
model_dir="bigvgan22HZ"

echo "下载 bigvgan22HZ 到 $model_dir ..."
hf download nvidia/bigvgan_v2_22khz_80band_256x --local-dir "$model_dir" || {
        echo "下载 bigvgan22HZ 失败"
        exit 1
}

# 定义 URL 和文件路径
mandarin_url="https://box.nju.edu.cn/d/8b66d18c1b624ce5b3a8/files/?p=%2Fdata%2Fmandarin.csv&dl=1"
cantonese_url="https://box.nju.edu.cn/d/8b66d18c1b624ce5b3a8/files/?p=%2Fdata%2Fcantonese.csv&dl=1"
mel_data_url="https://box.nju.edu.cn/d/8b66d18c1b624ce5b3a8/files/?p=%2Fdata%2Fmel_data.zip&dl=1"
wav_data_url="https://box.nju.edu.cn/d/8b66d18c1b624ce5b3a8/files/?p=%2Fdata%2Fwav_data.zip&dl=1"

all_data_dir="./data"
wav_data_dir="./data/wav.zip"
mel_data_dir="./data/mel.zip"
cantonese_file="./data/cantonese.csv"
mandarin_file="./data/mandarin.csv"

#删除原来的文件避免因为下载中断导致异常
rm -rf ./data

# 创建数据目录
mkdir -p "$all_data_dir" || { echo "无法创建目录 $all_data_dir"; exit 1; }

# 根据 download_wav 设置 URL 和文件列表
if [ "$download_wav" = "True" ]; then
    echo "下载语音数据"
    url_list=("$mandarin_url" "$cantonese_url" "$wav_data_url")
    file_list=("$mandarin_file" "$cantonese_file" "$wav_data_dir")
else
    echo "下载梅尔频谱数据"
    url_list=("$mandarin_url" "$cantonese_url" "$mel_data_url")
    file_list=("$mandarin_file" "$cantonese_file" "$mel_data_dir")
fi

# 下载和解压文件
for idx in {0..2}; do
    if [ -f "${file_list[idx]}" ]; then
        echo "${file_list[idx]} 已存在，跳过下载"
    else
        echo "下载 ${url_list[idx]} 到 ${file_list[idx]} ..."
        curl -L -o "${file_list[idx]}" "${url_list[idx]}" || { echo "下载 ${url_list[idx]} 失败"; exit 1; }
    fi
    
    if [[ "${file_list[idx]}" == *.zip ]]; then
        # 获取 ZIP 文件名前缀（去掉 .zip）
        #unzip_dir="$all_data_dir/$(basename "${file_list[idx]}" .zip)"
        
        if [ -f "${file_list[idx]}" ]; then
            #echo "解压 ${file_list[idx]} 到 $unzip_dir ..."
            #mkdir -p "$unzip_dir" || { echo "无法创建解压目录 $unzip_dir"; exit 1; }
            unzip -o "${file_list[idx]}" -d "./data" || { echo "解压 ${file_list[idx]} 失败"; exit 1; }
            rm -f "${file_list[idx]}" || { echo "删除 ${file_list[idx]} 失败"; exit 1; }
        else
            echo "${file_list[idx]} 不存在，跳过解压"
        fi
    fi
done

echo "所有文件处理完成"