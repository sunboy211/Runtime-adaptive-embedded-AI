import os
import io
import pandas as pd
import soundfile as sf

data_dir = r"F:\master thesis\demand\data"
output_dir = r"F:\master thesis\demand_extracted_wavs"
os.makedirs(output_dir, exist_ok=True)

print("Starting local offline extraction of 16k Parquet noise files...")

for filename in os.listdir(data_dir):
    if filename.startswith("16k") and filename.endswith(".parquet"):
        file_path = os.path.join(data_dir, filename)
        print(f" Processing file: {filename}")

        df = pd.read_parquet(file_path)

        for idx, row in df.iterrows():
            scene_path = row['name']
            scene_folder = scene_path.split('/')[0]

            # 直接获取音频的原始二进制字节流 (bytes)
            audio_bytes = row['audio']

            scene_output_dir = os.path.join(output_dir, scene_folder)
            os.makedirs(scene_output_dir, exist_ok=True)

            wav_name = os.path.basename(scene_path).replace('.ogg', '.wav')
            wav_output_path = os.path.join(scene_output_dir, wav_name)

            try:
                # 使用 io.BytesIO 把内存中的字节流伪装成文件，让 soundfile 去读取和解码
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

                # 强制写入本地为标准的 16kHz WAV 文件
                sf.write(wav_output_path, audio_data, sample_rate)
            except Exception as e:
                print(f" Warning: Failed to decode {scene_path}, error: {e}")

print(f"\n Done! All 16k noises are now standard WAV files in: {output_dir}")
