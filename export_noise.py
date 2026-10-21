import os
import io
import pandas as pd
import soundfile as sf

# 设置你本地存放 16k parquet 文件的绝对路径（根据你的截图）
data_dir = r"F:\master thesis\demand\data"
output_dir = r"F:\master thesis\demand_extracted_wavs"
os.makedirs(output_dir, exist_ok=True)

print("Starting local offline extraction of 16k Parquet noise files...")

# 遍历文件夹里所有剩下的 16k parquet 文件
for filename in os.listdir(data_dir):
    if filename.startswith("16k") and filename.endswith(".parquet"):
        file_path = os.path.join(data_dir, filename)
        print(f"📖 Processing file: {filename}")

        # 读取 parquet 数据集
        df = pd.read_parquet(file_path)

        for idx, row in df.iterrows():
            # 1. 获取噪声场景名称（例如 'DKITCHEN_16k/ch01_00.ogg'）
            scene_path = row['name']
            scene_folder = scene_path.split('/')[0]  # 提取出 'DKITCHEN_16k'

            # 2. 直接获取音频的原始二进制字节流 (bytes)
            audio_bytes = row['audio']

            # 3. 创建对应的输出文件夹（按场景分类，如 DKITCHEN_16k, OOFFICE_16k）
            scene_output_dir = os.path.join(output_dir, scene_folder)
            os.makedirs(scene_output_dir, exist_ok=True)

            # 4. 命名并保存为标准的 WAV
            wav_name = os.path.basename(scene_path).replace('.ogg', '.wav')
            wav_output_path = os.path.join(scene_output_dir, wav_name)

            try:
                # 使用 io.BytesIO 把内存中的字节流伪装成文件，让 soundfile 去读取和解码
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

                # 强制写入本地为标准的 16kHz WAV 文件
                sf.write(wav_output_path, audio_data, sample_rate)
            except Exception as e:
                print(f"⚠️ Warning: Failed to decode {scene_path}, error: {e}")

print(f"\n Done! All 16k noises are now standard WAV files in: {output_dir}")
