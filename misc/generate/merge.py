# 将多个 csv file 合并，audio 列重新编号
import pandas as pd


def merge_csv_files(file_list, output_file):
    """
    合并多个 CSV 文件，并重新编号 audio 列。
    
    :param file_list: 包含要合并的 CSV 文件路径的列表
    :param output_file: 合并后的输出文件路径
    """
    combined_data = []
    current_audio_number = 1

    for file in file_list:
        df = pd.read_csv(file)
        # 确保 audio 列存在
        if 'audio' in df.columns:
            df['audio'] = current_audio_number + df.index
            combined_data.append(df)
            current_audio_number += len(df)

    # 合并所有数据
    merged_df = pd.concat(combined_data, ignore_index=True)
    
    # 保存到新的 CSV 文件
    merged_df.to_csv(output_file, index=False)
    print(f"合并完成，保存到 {output_file}")

if __name__ == "__main__":
    # 示例文件列表
    files_to_merge = [
        'sentences.csv',
        'conv.csv',
    ]
    
    output_file = 'merged_output.csv'
    
    merge_csv_files(files_to_merge, output_file)