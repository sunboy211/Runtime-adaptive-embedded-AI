# Note: this is dynamic quantization(post-training quantization)
# STM32 only support static quantization
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.ao.quantization as quantization
from models import TinyAudioGRU, AudioDenoisingDataset, eval_collate_fn, evaluate_audio_quality_dnsmos


def evaluate_ptq_quantization(eval_test_loader, evaluate_dnsmos_fn):
    cpu_device = torch.device("cpu")

    # 加载原始 FP32 大模型
    model_fp32 = TinyAudioGRU(input_dim=256, hidden_dim=128, num_layers=3)
    model_fp32.load_state_dict(torch.load("model_large_best.pth", map_location=cpu_device))
    model_fp32.eval()

    # 将 GRU 和 Linear 层权重压为 INT8
    #torch.backends.quantized.engine = 'fbgemm'

    model_int8 = quantization.quantize_dynamic(
        model_fp32,
        {nn.GRU, nn.Linear},  # 指定需要量化的核心计算层
        dtype=torch.qint8
    )

    torch.save(model_fp32.state_dict(), "temp_fp32.pth")
    torch.save(model_int8.state_dict(), "temp_int8.pth")

    size_fp32 = os.path.getsize("temp_fp32.pth") / 1024.0
    size_int8 = os.path.getsize("temp_int8.pth") / 1024.0

    print("==================================================")
    print(f"【体积对比】原始 FP32 模型: {size_fp32:.2f} KB ({size_fp32 / 1024:.2f} MB)")
    print(f"【体积对比】量化 INT8 模型: {size_int8:.2f} KB ({size_int8 / 1024:.2f} MB)")
    print(f"【压缩收益】体积直接缩减了: {(1 - size_int8 / size_fp32) * 100:.1f}%")
    print("==================================================\n")

    # 清理临时文件
    os.remove("temp_fp32.pth")
    os.remove("temp_int8.pth")

    print("开始评估 INT8 量化大模型的 DNSMOS 听感得分...")
    evaluate_dnsmos_fn(model_int8, eval_test_loader, cpu_device, "LARGE")

    return model_int8

if __name__ == '__main__':
    NOISY_DATA_DIR = "F:/master thesis/processed_dataset/train_snr_0"
    CLEAN_DATA_DIR = "F:/master thesis/dev-clean/LibriSpeech/dev-clean"
    raw_clean_files = sorted(glob.glob(os.path.join(CLEAN_DATA_DIR, "**/*.flac"), recursive=True))
    base_clean_files = raw_clean_files[:200]

    # 直接构建强绑定的元组列表
    paired_dataset_files = []
    for clean_path in base_clean_files:
        base_name = os.path.splitext(os.path.basename(clean_path))[0]
        target_noisy_name = f"{base_name}.wav"
        target_noisy_path = os.path.join(NOISY_DATA_DIR, target_noisy_name)

        if os.path.exists(target_noisy_path):
            paired_dataset_files.append((target_noisy_path, clean_path))

    print(f"Total number of successfully and strictly bound and aligned audio pairs: {len(paired_dataset_files)}")
    for i in range(min(2, len(paired_dataset_files))):
        n_p, c_p = paired_dataset_files[i]
        print(f"bound pairs [{i}]: {os.path.basename(n_p)} 🔗 {os.path.basename(c_p)}")
    print("------------------------------------------\n")

    split_idx = int(len(paired_dataset_files) * 0.8)
    val_pairs = paired_dataset_files[split_idx:]
    val_dataset = AudioDenoisingDataset(val_pairs)
    eval_test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)
    quantized_model = evaluate_ptq_quantization(eval_test_loader, evaluate_audio_quality_dnsmos)