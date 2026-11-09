import os
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import glob
import numpy as np
import soundfile as sf
from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score as dnsmos

class AudioDenoisingDataset(Dataset):
    def __init__(self, paired_files, target_sr=16000):
        self.paired_files = paired_files
        self.target_sr = target_sr
        # waveform--STFT--> spectrogram
        self.n_fft = 512
        self.win_length = 512
        self.hop_length = 256
        self.window = torch.hann_window(self.win_length)

    def __len__(self):
        return len(self.paired_files)

    def _process_audio(self, file_path):
        """ 使用 soundfile 原生读取，在 NumPy 层面严格强制单声道 """
        # 直接从硬盘读取为 NumPy 矩阵
        raw_data, sr = sf.read(file_path)

        # 在安全稳定的 NumPy 层面直接压平多通道，杜绝转置引发的维度颠倒灾难
        if raw_data.ndim > 1:
            # 如果通道数在最后一维 (如 [Samples, Channels])，沿着轴 1 求均值
            raw_data = np.mean(raw_data, axis=1)

        # 此时 raw_data 必然是纯单声道的 [Samples] 形状，安全将其转为 Tensor 并规范为 [1, Samples]
        waveform = torch.from_numpy(raw_data).float().unsqueeze(0)

        # 重采样与单声道对齐
        if sr != self.target_sr:
            resample = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resample(waveform)

        complex_spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,  # 保持与 torchaudio 默认一致的 center=True
            return_complex=True
        )  # 输出形状: [Freq_Bins(257), Time_Frames]

        magnitude = torch.abs(complex_spec)  # 等价于之前的 power=1.0 变换
        phase = torch.angle(complex_spec)
        magnitude = magnitude.transpose(0, 1)
        phase = phase.transpose(0, 1)

        mag_256 = magnitude[:, :256]
        phase_256 = phase[:, :256]

        return mag_256, phase_256

    def __getitem__(self, idx):
        # 通过唯一的索引直接解包成对的路径，即使 Shuffle 打乱，它们也生死相随
        noisy_path, clean_path = self.paired_files[idx]
        noisy_mag, noisy_phase = self._process_audio(noisy_path)
        clean_mag, _ = self._process_audio(clean_path)

        assert noisy_mag.shape == clean_mag.shape, \
            f"物理长度冲突错误！\n带噪音频: {noisy_path} (形状: {noisy_mag.shape})\n干净音频: {clean_path} (形状: {clean_mag.shape})"

        # 返回三个张量：带噪幅度、带噪相位、干净幅度
        return noisy_mag, noisy_phase, clean_mag


def denoising_collate_fn(batch):
    noisy_mags, _, clean_mags = zip(*batch)
    # [Batch, Max_Time_Frames, 256]
    noisy_padded = pad_sequence(noisy_mags, batch_first=True, padding_value=0.0)
    clean_padded = pad_sequence(clean_mags, batch_first=True, padding_value=0.0)
    return noisy_padded, clean_padded


def eval_collate_fn(batch):
    # 确保 batch 里面只有 1 条音频数据（因为 eval_test_loader 的 batch_size=1）
    noisy_mag, noisy_phase, clean_mag = batch[0]

    # 增加 Batch 维度，变成模型和评测所需要的 [1, Time_Frames, 256]
    return noisy_mag.unsqueeze(0), noisy_phase.unsqueeze(0), clean_mag.unsqueeze(0)


class TinyAudioGRU(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, num_layers=2):
        super(TinyAudioGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.h_state = None

    def forward(self, x, h_state=None):
        noisy_input = x
        x, next_h_state = self.gru(x, h_state)
        mask = self.sigmoid(self.fc(x))
        return noisy_input * mask, next_h_state


def evaluate_audio_quality_dnsmos(model, test_loader, device):
    """
    使用 DNSMOS 指标评估降噪模型（SIG、BAK、OVRL）
    Speech signal quality, Background noise quality, Overall speech quality
    """
    model.eval()

    scores = {
        'noisy_sig': 0.0, 'denoised_sig': 0.0,
        'noisy_bak': 0.0, 'denoised_bak': 0.0,
        'noisy_ovrl': 0.0, 'denoised_ovrl': 0.0
    }
    count = 0

    # 逆谱变换器
    istft_transform = T.InverseSpectrogram(n_fft=512, win_length=512, hop_length=256).to(device)
    #griffin_lim = T.GriffinLim(n_fft=512, win_length=512, hop_length=256, power=1.0, n_iter=32).to(device)

    # 将 256 维补齐为 257 维，并还原回时域波形 [1, Samples]
    def to_waveform_by_phase(magnitude_tensor, phase_tensor):
        pad_mag = torch.zeros((*magnitude_tensor.shape[:-1], 1), device=magnitude_tensor.device)
        pad_phase = torch.zeros((*phase_tensor.shape[:-1], 1), device=phase_tensor.device)

        full_mag = torch.cat([magnitude_tensor, pad_mag], dim=-1)
        full_phase = torch.cat([phase_tensor, pad_phase], dim=-1)

        complex_spec = torch.polar(full_mag, full_phase).transpose(1, 2)
        return istft_transform(complex_spec)
        """
        pad_mag = magnitude_tensor[:, :, -1].unsqueeze(-1)
        full_mag = torch.cat([magnitude_tensor, pad_mag], dim=-1)
        pad_phase = phase_tensor[:, :, -1].unsqueeze(-1)
        full_phase = torch.cat([phase_tensor, pad_phase], dim=-1)
        complex_spec = torch.polar(full_mag, full_phase).transpose(1, 2)
        return istft_transform(complex_spec)
        """

    print("\n 开始通过 Microsoft DNSMOS 模型计算测试集指标...")
    print("提示：如果是首次运行，torchmetrics 会自动从官方服务器下载 ONNX 模型权重文件。")

    with torch.no_grad():
        for noisy_features, noisy_phase, _ in test_loader:
            model.h_state = None
            noisy_features = noisy_features.to(device)
            noisy_phase = noisy_phase.to(device)

            # 前向传播，得到降噪后的频谱
            denoised_features, _ = model(noisy_features, h_state=None)
            # 补偿性 保留 15% 的自然底噪
            if MODEL_TYPE == "SMALL":
                denoised_features = 0.85 * denoised_features + 0.15 * noisy_features

            wav_noisy = to_waveform_by_phase(noisy_features, noisy_phase)
            wav_denoised = to_waveform_by_phase(denoised_features, noisy_phase)
            """
            print(f"张量的几何维度（几维）: {wav_noisy.ndim} 维")
            print(f"张量的精确形状 (Shape)  : {list(wav_noisy.shape)}")
            print(f"张量的几何维度（几维）: {wav_denoised.ndim} 维")
            print(f"张量的精确形状 (Shape)  : {list(wav_denoised.shape)}")
            """
            wav_noisy_1d = wav_noisy.squeeze()
            wav_denoised_1d = wav_denoised.squeeze()
            if wav_noisy_1d.ndim == 0 or wav_noisy_1d.shape[0] == 0:
                print(f" 跳过第 {count + 1} 条音频：长度异常为 0")
                continue

            min_samples = 64000
            if wav_noisy_1d.shape[0] < min_samples:
                repeats = int(np.ceil(min_samples / wav_noisy_1d.shape[0]))
                wav_noisy_1d = wav_noisy_1d.repeat(repeats)
                wav_denoised_1d = wav_denoised_1d.repeat(repeats)
            try:
                res_noisy = dnsmos(wav_noisy_1d, fs=16000, personalized=False, device=device)
                res_denoised = dnsmos(wav_denoised_1d, fs=16000, personalized=False, device=device)

                # 根据源码对应的索引提取对应分数的标量值：
                # 索引 1 是 SIG, 索引 2 是 BAK, 索引 3 是 OVRL
                scores['noisy_sig'] += res_noisy[1].item()
                scores['denoised_sig'] += res_denoised[1].item()
                scores['noisy_bak'] += res_noisy[2].item()
                scores['denoised_bak'] += res_denoised[2].item()
                scores['noisy_ovrl'] += res_noisy[3].item()
                scores['denoised_ovrl'] += res_denoised[3].item()
                count += 1

            except Exception as e:
                import traceback
                print(f"\n [ERROR] 运行至第 {count + 1} 条音频时崩了！")
                print(f"当前一维输入形状 wav_noisy_1d shape: {list(wav_noisy_1d.shape)}")
                traceback.print_exc()
                continue
    # 计算并输出平均分
    if count > 0:
        avg_scores = {k: v / count for k, v in scores.items()}
        print(f"\n DNSMOS 评估完成！共成功处理 {count} 条音频。")
        print(f"==================================================")
        print(
            f"【带噪基准】 SIG: {avg_scores['noisy_sig']:.3f} | BAK: {avg_scores['noisy_bak']:.3f} | OVRL: {avg_scores['noisy_ovrl']:.3f}")
        print(
            f"【模型降噪】 SIG: {avg_scores['denoised_sig']:.3f} | BAK: {avg_scores['denoised_bak']:.3f} | OVRL: {avg_scores['denoised_ovrl']:.3f}")
        print(f"==================================================")
        return avg_scores
    else:
        print(" 未成功评估任何音频，请检查输入或 dnsmos 初始化是否正确。")
        return None

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前检测到的硬件加速设备: {device}")

    # 大模型
    NOISY_DATA_DIR = "F:/master thesis/processed_dataset/train_snr_0"
    MODEL_TYPE = "LARGE"

    # 小模型
    #NOISY_DATA_DIR = "F:/master thesis/processed_dataset/train_snr_10"
    #MODEL_TYPE = "SMALL"

    CLEAN_DATA_DIR = "F:/master thesis/dev-clean/LibriSpeech/dev-clean"

    # 严格复刻合成逻辑获取干净音频
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

    print(f"\n--- 物理对齐绑定检查 ---")
    print(f"成功严格绑定并对齐的音频对总数: {len(paired_dataset_files)}")
    for i in range(min(2, len(paired_dataset_files))):
        n_p, c_p = paired_dataset_files[i]
        print(f"绑定对 [{i}]: {os.path.basename(n_p)} 🔗 {os.path.basename(c_p)}")
    print("------------------------------------------\n")

    split_idx = int(len(paired_dataset_files) * 0.8)
    train_pairs = paired_dataset_files[:split_idx]
    val_pairs = paired_dataset_files[split_idx:]

    train_dataset = AudioDenoisingDataset(train_pairs)
    val_dataset = AudioDenoisingDataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=denoising_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=denoising_collate_fn)
    eval_test_loader= DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

    X_noisy, Y_clean = next(iter(train_loader))
    print("----- 恭喜！成对数据流打通成功！-------")
    print("输入带噪 Batch 形状 (X):", X_noisy.shape)
    print("目标干净 Batch 形状 (Y):", Y_clean.shape)


    input_dim = 256
    # 动态初始化和指定当前要训练的模型与保存路径
    if MODEL_TYPE == "LARGE":
        current_model = TinyAudioGRU(input_dim, hidden_dim=128, num_layers=3).to(device)
        current_model_path = "model_large_best.pth"
        print(" 当前激活任务：训练【大模型】(hidden_dim=128, layers=3)")
    else:
        current_model = TinyAudioGRU(input_dim, hidden_dim=16, num_layers=1).to(device)
        current_model_path = "model_small_best.pth"
        print(" 当前激活任务：训练【小模型】(hidden_dim=16, layers=1)")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(current_model.parameters(), lr=0.001)

    num_epochs = 40 # 针对200对数据
    best_val_loss = float('inf')
    print(f"\n 开始在 GPU ({device}) 上执行训练任务...")

    for epoch in range(num_epochs):
        current_model.train()
        total_loss = 0.0

        for noisy_features, clean_targets in train_loader:

            noisy_features = noisy_features.to(device)
            clean_targets = clean_targets.to(device)

            optimizer.zero_grad()

            denoised_outputs, _ = current_model(noisy_features, h_state=None)

            loss = criterion(denoised_outputs, clean_targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_train_loss = total_loss / len(train_loader)

        # --- 验证阶段 (Validation) ---
        current_model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs, _ = current_model(batch_x, h_state=None)
                loss = criterion(outputs, batch_y)

                # 直接累加 Batch 的平均 Loss，后续除以 Batch 总数，与训练集完美对齐
                total_val_loss += loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)

        # --- 打印日志与保存最优模型 ---
        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Spectral MSE Loss: {epoch_train_loss:.6f} | "
              f"Val Spectral MSE Loss: {epoch_val_loss:.6f}")
        # 只有在验证集 Loss 创下新低时才保存模型（防止过拟合的灵魂所在）
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(current_model.state_dict(), current_model_path)
            print(f" 发现更好的模型！验证集 Loss 降至 {best_val_loss:.6f}，权重已保存。")

    print(" 训练与验证完全结束！")
    current_model.load_state_dict(torch.load(current_model_path))
    evaluate_audio_quality_dnsmos(current_model, eval_test_loader, device)