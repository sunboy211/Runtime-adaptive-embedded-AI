import os
import glob
import math
import numpy as np
import soundfile as sf

NOISE_DIR = r"F:\master thesis\demand_extracted_wavs"
CLEAN_DIR = r"F:\master thesis\dev-clean\LibriSpeech\dev-clean"
OUTPUT_DIR = r"F:\master thesis\processed_dataset"

#sound-noise ratio; For small model:high SNR;For big model:low SNR
SNR_LEVELS = [-5, 0, 5, 10, 15]
TARGET_SR = 16000 #16kHz

for snr in SNR_LEVELS:
    os.makedirs(os.path.join(OUTPUT_DIR, f"train_snr_{snr}"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test_adaptive_mutation"), exist_ok=True)

def calculate_snr_scale(clean_data, noise_data, target_snr):
    """
    根据目标SNR计算噪声的缩放系数 (能量对齐)
    """
    # 计算均方根能量 (RMS)
    clean_rms = np.sqrt(np.mean(clean_data ** 2)) + 1e-8
    noise_rms = np.sqrt(np.mean(noise_data ** 2)) + 1e-8

    # SNR = 20 * log10(clean_rms / (noise_rms * scale))
    scale = (clean_rms / noise_rms) / (10 ** (target_snr / 20.0))
    return scale


def start_mixing():
    clean_files = sorted(glob.glob(os.path.join(CLEAN_DIR, "**/*.flac"), recursive=True))
    noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "**/*.wav"), recursive=True))

    if not clean_files:
        print(f"Error：在路径下未找到 .flac 文件，请检查：{CLEAN_DIR}")
        return
    if not noise_files:
        print(f"Error：在路径下未找到噪声 .wav 文件，请检查：{NOISE_DIR}")
        return

    print(f"Loaded {len(clean_files)} clean files and {len(noise_files)} noise files.")
    print("Starting data augmentation and mixing pipeline...")

    # 为了防止文件对不齐，这里用循环进行混合
    for idx, clean_path in enumerate(clean_files[:200]):  # 先拿前200条跑通全流程
        clean_data, sr = sf.read(clean_path)
        if sr != TARGET_SR:
            print(f" 跳过：索引 {idx} 的文件 {os.path.basename(clean_path)} 采样率是 {sr}Hz，不是 16000Hz！")
            continue  # 确保采样率一致

        # 随机或轮询挑选一个噪声文件
        noise_path = noise_files[idx % len(noise_files)]
        noise_data, n_sr = sf.read(noise_path)

        # 2. 长度对齐处理 (时域裁剪或填充)
        if len(noise_data) < len(clean_data):
            # 如果噪声太短，通过循环平铺（Tile）将其加长
            repeats = math.ceil(len(clean_data) / len(noise_data))
            noise_data = np.tile(noise_data, repeats)
        # 截取与人声完全相同长度的噪声
        noise_slice = noise_data[:len(clean_data)]

        # 3. 生成常规的固定 SNR 训练集
        for snr in SNR_LEVELS:
            scale = calculate_snr_scale(clean_data, noise_slice, target_snr=snr)
            mixed_data = clean_data + scale * noise_slice

            # 防止音频叠加后爆音（Clipping），进行归一化
            max_val = np.max(np.abs(mixed_data))
            if max_val > 1.0:
                mixed_data = mixed_data / max_val

            base_name = os.path.splitext(os.path.basename(clean_path))[0]
            file_name = f"{base_name}.wav"

            out_path = os.path.join(OUTPUT_DIR, f"train_snr_{snr}", file_name)
            sf.write(out_path, mixed_data, TARGET_SR)

        # 4. 手工合成一段“噪声突变”的音频 为了MAPE-K loop
        if idx < 10:  # 生成 10 条突变测试音频足够了
            half_len = len(clean_data) // 2
            # 前半段：15dB（安静，对应小模型常态运作）
            scale_quiet = calculate_snr_scale(clean_data[:half_len], noise_slice[:half_len], target_snr=15)
            # 后半段：突变到 -5dB（极度嘈杂，测试状态机能否在几毫秒内测出能量变化并切到大模型）
            scale_loud = calculate_snr_scale(clean_data[half_len:], noise_slice[half_len:], target_snr=-5)

            mutation_noise = np.concatenate([scale_quiet * noise_slice[:half_len], scale_loud * noise_slice[half_len:]])
            mutated_audio = clean_data + mutation_noise

            # 归一化
            if np.max(np.abs(mutated_audio)) > 1.0:
                mutated_audio = mutated_audio / np.max(np.abs(mutated_audio))

            mutation_out_path = os.path.join(OUTPUT_DIR, "test_adaptive_mutation", f"mutation_test_{idx}.wav")
            sf.write(mutation_out_path, mutated_audio, TARGET_SR)

    print(f" Dataset preparation complete! Look inside: {OUTPUT_DIR}")


if __name__ == "__main__":
    start_mixing()