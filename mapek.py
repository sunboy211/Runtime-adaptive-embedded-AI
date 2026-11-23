from models import TinyAudioGRU, AudioDenoisingDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import glob
import soundfile as sf

class AcousticWorldModel(nn.Module):
    def __init__(self):
        super(AcousticWorldModel, self).__init__()

    def estimate_noise_severity(self, active_model, current_h_state, init_frame_tensor, lookahead_ms=48, frame_size_ms=16):
        rollout_steps = int(lookahead_ms / frame_size_ms)

        with torch.no_grad():
            # 保护真机实时记忆
            virtual_h = current_h_state.clone() if current_h_state is not None else None

            # 初始虚拟输入采用当前帧的真实特征，而非盲目的全 0
            virtual_input = init_frame_tensor.clone()
            init_energy = torch.mean(init_frame_tensor ** 2).item()
            future_noise_energies = []
            estimated_noise_floor = 0.0

            for step in range(rollout_steps):
                # 脑内自回归空转
                predicted_clean, virtual_h = active_model(virtual_input, h_state=virtual_h)

                clean_energy = torch.mean(predicted_clean ** 2).item()

                if step == 0:
                    # 剥离人声：总能量 - 预测纯净语音能量 = 纯噪声能量
                    estimated_noise_floor = max(0.0, init_energy - clean_energy)
                predicted_noise_energy = estimated_noise_floor * (0.92 ** step)
                future_noise_energies.append(predicted_noise_energy)

                # 自回归特征传递
                virtual_input = init_frame_tensor.detach()

        return estimated_noise_floor, np.mean(future_noise_energies)


class MapekAncController:
    def __init__(self, model_small, model_large, world_model, device):
        self.model_small = model_small
        self.model_large = model_large
        self.world_model = world_model
        self.device = device

        self.STATE_SMALL = 0
        self.STATE_LARGE = 1
        self.current_state = self.STATE_SMALL

        self.NOISE_ENTER_THRESHOLD = 1.5
        self.WAKE_CHECK_THRESHOLD = 1.0
        self.NOISE_EXIT_THRESHOLD = 0.8

        self.ATTACK_FRAMES = 3    # 连续 3 帧 (48ms) 噪声超标才升档
        self.HOLD_TIME_MS = 800
        self.COOLDOWN_TIME_MS = 480  # 降档后冷静 480ms (30帧)，防止乒乓抖动
        self.FADE_TIME_MS = 96    # Raised from 32ms to 96ms for the low-power single-model output smooth smoothing.
        # A 32ms window is too abrupt for zero-state initialization and causes pops.

        self.hold_max_frames = int(self.HOLD_TIME_MS / 16)
        self.cooldown_max_frames = int(self.COOLDOWN_TIME_MS / 16)
        self.fade_max_frames = int(self.FADE_TIME_MS / 16)

        self.attack_counter = 0
        self.hold_counter = 0
        self.cooldown_counter = 0
        self.ema_noise_energy = 0.0
        self.h_small = None
        self.h_large = None

    def reset_stream_states(self):
        self.h_small = None
        self.h_large = None
        self.current_state = self.STATE_SMALL
        self.attack_counter = 0
        self.hold_counter = 0
        self.cooldown_counter = 0
        self.ema_noise_energy = 0.0

    def _run_small_model(self, frame_tensor):
        """统一小模型推理与直通混音逻辑，消除过渡期结束时的底噪阶跃爆音"""
        out_small, self.h_small = self.model_small(frame_tensor, self.h_small)
        return 0.85 * out_small + 0.15 * frame_tensor

    def process_audio_stream(self, noisy_spec):
        """
        输入来自 Dataset 提取的频域特征矩阵
        noisy_spec 形状: [Time_Frames, 256]
        """
        num_frames = noisy_spec.shape[0]

        output_spec = torch.zeros_like(noisy_spec, device=self.device)
        state_history = np.zeros(num_frames)
        energy_history = np.zeros(num_frames)
        noise_trend_history = np.zeros(num_frames)

        fade_idx = 0
        is_fading = False
        fade_direction = None

        # 全局流式循环（以 16ms 帧为单位向前推进）
        for f in range(num_frames):
            # 将当前 16ms 的一帧截取并重塑为标准的 3D Tensor 形状 [1, 1, 256]
            frame_tensor = noisy_spec[f].view(1, 1, 256).to(self.device)

            # --- 1. M (Monitor) 阶段：计算当前帧基础能量（极低开销） ---
            current_energy = torch.mean(frame_tensor ** 2).item()

            current_noise_est = 0.0
            future_noise_trend = 0.0

            # --- 2. P (Plan) 阶段：极致节能状态机与条件式算力唤醒 ---
            if self.current_state == self.STATE_SMALL:
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1
                # 极致节能核心门控：当前能量大于唤醒检查阈值时，才调用世界模型
                if current_energy > self.WAKE_CHECK_THRESHOLD:
                        current_noise_est, future_noise_trend = self.world_model.estimate_noise_severity(
                            self.model_small, self.h_small, frame_tensor
                        )
                self.ema_noise_energy = 0.7 * self.ema_noise_energy + 0.3 * future_noise_trend

                # 满足双重超标条件，决定升级为大模型
                if self.cooldown_counter == 0 and self.ema_noise_energy > self.NOISE_ENTER_THRESHOLD:
                    self.attack_counter += 1
                    if self.attack_counter >= self.ATTACK_FRAMES:
                        self.current_state = self.STATE_LARGE
                        is_fading = True
                        fade_idx = 0
                        fade_direction = 'S2B'
                        self.h_large = None
                        self.hold_counter = self.hold_max_frames
                        self.attack_counter = 0
                else:
                    self.attack_counter = max(0, self.attack_counter-1)

            elif self.current_state == self.STATE_LARGE:
                # 在大模型模式下，同样利用大小模型差异或简化能量估算当前噪声
                if f % 2 == 0:
                    with torch.no_grad():
                        temp_clean, _ = self.model_small(frame_tensor, self.h_small)
                        clean_e = torch.mean(temp_clean ** 2).item()
                        current_noise_est = max(0.0, current_energy - clean_e)
                        self.ema_noise_energy = 0.7 * self.ema_noise_energy + 0.3 * current_noise_est

                # future_noise_trend = self.ema_noise_energy

                if self.ema_noise_energy >= self.NOISE_EXIT_THRESHOLD:
                    self.hold_counter = self.hold_max_frames
                else:
                    self.hold_counter -= 1

                # 噪声真正消退并维持了 HOLD_TIME 后，切回小模型
                if self.hold_counter <= 0:
                    self.current_state = self.STATE_SMALL
                    is_fading = True
                    fade_idx = 0
                    fade_direction = 'B2S'
                    self.h_small = None  # 清除过期记忆，配合 96ms 交叉淡入实现无爆音重启
                    self.cooldown_counter = self.cooldown_max_frames

            # --- 3. Execute (E): 交叉淡入淡出执行 ---
            with torch.no_grad():
                if is_fading:
                    fade_idx += 1
                    w_new = fade_idx / self.fade_max_frames

                    out_small = self._run_small_model(frame_tensor)
                    out_big, self.h_large = self.model_large(frame_tensor, self.h_large)

                    if fade_direction == 'S2B':
                        output_frame = (1.0 - w_new) * out_small + w_new * out_big
                    else:
                        output_frame = (1.0 - w_new) * out_big + w_new * out_small

                    if fade_idx >= self.fade_max_frames:
                        is_fading = False
                else:
                    if self.current_state == self.STATE_LARGE:
                        output_frame, self.h_large = self.model_large(frame_tensor, self.h_large)
                    else:
                        output_frame = self._run_small_model(frame_tensor)

            # 转换回 256 维向量存入输出矩阵（移回 CPU 写入）
            output_spec[f] = output_frame.view(256)
            state_history[f] = self.current_state
            energy_history[f] = current_energy
            noise_trend_history[f] = self.ema_noise_energy

        history = {
            "state": state_history,
            "energy": energy_history,
            "trend": noise_trend_history,
            "enter_threshold": self.NOISE_ENTER_THRESHOLD,
            "exit_threshold": self.NOISE_EXIT_THRESHOLD,
            "wake_threshold": self.WAKE_CHECK_THRESHOLD
        }
        return output_spec.cpu(), history

def extract_features_aligned(wav_path, dataset_helper):
    mag_256, phase_256 = dataset_helper._process_audio(wav_path)
    return mag_256, phase_256


def reconstruct_and_save_aligned(output_spec, phase_256, save_path, target_sr=16000):
    mag_256 = output_spec.detach().cpu().float()
    phase_256 = phase_256.detach().cpu().float()

    num_frames = mag_256.shape[0]
    zero_bin = torch.zeros((num_frames, 1), dtype=torch.float32)

    mag_257 = torch.cat([mag_256, zero_bin], dim=1)
    phase_257 = torch.cat([phase_256, zero_bin], dim=1)

    mag_257 = mag_257.transpose(0, 1)
    phase_257 = phase_257.transpose(0, 1)

    complex_spec = torch.polar(mag_257, phase_257)

    cleaned_waveform = torch.istft(
        complex_spec,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512),
        center=True,
        return_complex=False
    )

    cleaned_np = cleaned_waveform.numpy()
    sf.write(save_path, cleaned_np, target_sr)
    return cleaned_np

def plot_mapek_analysis(noisy_wav_path, cleaned_wav, history, plot_save_path, sr=16000):
    """
    绘制四联对比图：带噪波形、降噪波形、频域能量与预测趋势、MAPE-K 状态切换轨迹
    """
    noisy_wav, _ = sf.read(noisy_wav_path)
    if noisy_wav.ndim > 1:
        noisy_wav = noisy_wav[:, 0]

    # 对齐波形长度
    min_len = min(len(noisy_wav), len(cleaned_wav))
    noisy_wav = noisy_wav[:min_len]
    cleaned_wav = cleaned_wav[:min_len]
    # time_wav = np.arange(min_len) / sr

    # 帧时间轴 (16ms/帧)
    num_frames = len(history["state"])
    time_frames = np.arange(num_frames) * 0.016

    VMIN, VMAX = -100, -20

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.2, 0.8]})

    # 原始带噪
    axes[0].specgram(noisy_wav, NFFT=512, Fs=16000, noverlap=256, cmap='magma', vmin=VMIN, vmax=VMAX)
    axes[0].set_ylabel("Freq (Hz)")
    axes[0].set_title(f"MAPE-K ANC: {os.path.basename(noisy_wav_path)}", fontsize=13, fontweight='bold')
    axes[0].text(0.015, 0.88, "Noisy Audio Spectrogram", transform=axes[0].transAxes,
              color="white", fontsize=10, fontweight="bold",
              bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
    axes[0].grid(True, alpha=0.3)

    # 降噪后
    axes[1].specgram(cleaned_wav, NFFT=512, Fs=16000, noverlap=256, cmap='magma', vmin=VMIN, vmax=VMAX)
    axes[1].set_ylabel("Freq (Hz)")
    axes[1].text(0.015, 0.88, "MAPE-K Denoised Audio Spectrogram", transform=axes[1].transAxes,
              color="white", fontsize=10, fontweight="bold",
              bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
    axes[1].grid(True, alpha=0.3)

    # 能量监测与世界模型预测趋势
    axes[2].plot(time_frames, history["energy"], label="Current Energy", color='#2c3e50', linewidth=1.2)
    axes[2].plot(time_frames, history["trend"], label="Noise Trend (EMA)", color='#e67e22', linestyle='--', linewidth=1.2)
    axes[2].axhline(history["enter_threshold"], color='#e74c3c', linestyle=':', linewidth=1.2, label=f" Enter Threshold ({history['enter_threshold']})")
    axes[2].axhline(history["exit_threshold"], color='#3498db', linestyle=':', linewidth=1.2, label=f" Exit Threshold ({history['exit_threshold']})")
    axes[2].axhline(history["wake_threshold"], color='#27ae60', linestyle=':', linewidth=1.0, label=f" Wake Threshold ({history['wake_threshold']:.2f})")
    axes[2].set_ylabel("Spectral Energy")
    axes[2].legend(loc="upper right", ncol=2)
    axes[2].grid(True, alpha=0.3)

    # 模型激活状态轨迹 (0=小模型, 1=大模型)
    axes[3].step(time_frames, history["state"], where='post', color='#8e44ad', linewidth=1.8)
    axes[3].fill_between(time_frames, history["state"], step='post', color='#8e44ad', alpha=0.25)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["Small Model", "Large Model"])
    axes[3].set_ylim(-0.2, 1.3)
    axes[3].set_xlabel("Time (seconds)", fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently used hardware device: {device}")
    Adapt_DATA_DIR="F:/master thesis/processed_dataset/test_adaptive_mutation"
    OUTPUT_DIR = os.path.join(Adapt_DATA_DIR, "mapek_outputs")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    model_large = TinyAudioGRU(input_dim=256, hidden_dim=128, num_layers=3).to(device)
    model_small = TinyAudioGRU(input_dim=256, hidden_dim=16, num_layers=1).to(device)
    world_model = AcousticWorldModel().to(device)
    model_small.load_state_dict(torch.load("model_small_best.pth", map_location=device))
    model_large.load_state_dict(torch.load("model_large_best.pth", map_location=device))

    model_small.eval()
    model_large.eval()
    world_model.eval()

    dataset_helper = AudioDenoisingDataset(Adapt_DATA_DIR)

    controller = MapekAncController(
        model_small=model_small,
        model_large=model_large,
        world_model=world_model,
        device=device
    )
    wav_files = glob.glob(os.path.join(Adapt_DATA_DIR, "*.wav"))
    print(f"Successfully retrieved {len(wav_files)} audio file(s) from the target path.")
    for i, wav_path in enumerate(wav_files):
        controller.reset_stream_states()
        file_name = os.path.basename(wav_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"\n[{i + 1}/{len(wav_files)}] Extracting features: {file_name}...")
        mag_256, phase_256 = extract_features_aligned(wav_path, dataset_helper)

        output_spec, history = controller.process_audio_stream(mag_256)

        save_path = os.path.join(OUTPUT_DIR, f"denoised_{file_name}")
        cleaned_wav= reconstruct_and_save_aligned(output_spec, phase_256, save_path)

        save_plot_path = os.path.join(PLOT_DIR, f"analysis_{base_name}.png")
        plot_mapek_analysis(wav_path, cleaned_wav, history, save_plot_path)

        large_ratio = np.mean(history["state"]) * 100
        print(f" Finish: {file_name} | Activation ratio of large models: {large_ratio:.1f}%")

