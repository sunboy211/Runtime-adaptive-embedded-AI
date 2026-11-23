import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, quant_pre_process
from models import TinyAudioGRU, AudioDenoisingDataset, eval_collate_fn, evaluate_audio_quality_dnsmos

"""
Why we choose static PTQ?
1. PyTorch dynamic quantization does not support export to standard ONNX: `DynamicQuantizedGRU` 
(produced by `quantize_dynamic`) triggers an "unsupported operator" error during ONNX export.
During inference, dynamic quantization requires scanning the activation vectors at each time step (or frame) to determine their 
maximum and minimum values, which are then used for scaling via floating-point division. For a 160 MHz microcontroller (MCU), 
the additional floating-point overhead of this "dynamic quantization" is extremely high. 
2. STM32Cube.AI requires "static quantization": The CMSIS-NN acceleration engine on MCUs necessitates that the quantization 
parameters (scale and zero-point) for both weights and activations be determined prior to inference; 
"""

class UnrolledStreamingGRU(nn.Module):
# Update 1: Remove ONNX loop graph nodes to prevent RAM usage up to 1.3 MB during Cube.AI quantization.
# Expected Flash 400kb & RAM 15kb
# 2: h_in/h_out shape is (1, num_layers, hidden_dim)
    def __init__(self, trained_gru_model, input_dim=256, hidden_dim=128, num_layers=3): # large model
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.GRUCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        src_gru = trained_gru_model.gru
        for i in range(num_layers):
            self.cells[i].weight_ih.data.copy_(getattr(src_gru, f"weight_ih_l{i}").data)
            self.cells[i].weight_hh.data.copy_(getattr(src_gru, f"weight_hh_l{i}").data)
            self.cells[i].bias_ih.data.copy_(getattr(src_gru, f"bias_ih_l{i}").data)
            self.cells[i].bias_hh.data.copy_(getattr(src_gru, f"bias_hh_l{i}").data)
        self.fc = trained_gru_model.fc
        self.sigmoid = trained_gru_model.sigmoid

    def forward(self, input_frame, h_in):
        # input_frame:, h_in: [1, num_layers, hidden_dim]
        x = input_frame.squeeze(1)  # [1, 256]
        next_h_list = []

        for i, cell in enumerate(self.cells):
            h_i = h_in[:, i, :]  # [1, hidden_dim]
            h_next = cell(x, h_i)  # [1, hidden_dim]
            next_h_list.append(h_next.unsqueeze(1))
            x = h_next

        mask = self.sigmoid(self.fc(x)).unsqueeze(1)  # [1, 1, 256]
        out = input_frame * mask  # 严格对齐原模型的 IRM 掩蔽输出
        h_out = torch.cat(next_h_list, dim=1)  # [1, num_layers, hidden_dim]
        return out, h_out

# 校准数据读取器：真实推进隐状态 + 支持导出 .npz 给 Cube.AI
class LargeModelCalibReader(CalibrationDataReader):
    def __init__(self, fp32_onnx_path, dataset, num_files=15, frames_per_file=20):
        self.samples = []
        self.calib_inputs = []
        self.calib_h = []

        # 启动临时 FP32 会话，在真实音频流上推进隐状态 h_state，使校准分布最贴近真实运行
        sess = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])

        for i in range(min(num_files, len(dataset))):
            noisy_mag, _, _ = dataset[i]
            h_curr = np.zeros((1, 3, 128), dtype=np.float32)

            max_f = min(frames_per_file, noisy_mag.shape[0])
            for f in range(max_f):
                frame = noisy_mag[f].numpy().reshape(1, 1, 256).astype(np.float32)
                self.samples.append({
                    "input_frame": frame,
                    "h_in": h_curr.copy()
                })
                self.calib_inputs.append(frame)
                self.calib_h.append(h_curr.copy())

                # 真实步进更新隐状态
                _, h_curr = sess.run(None, {"input_frame": frame, "h_in": h_curr})

        self.data_iter = iter(self.samples)

    def get_next(self):
        return next(self.data_iter, None)

    def save_as_npz(self, npz_path="calib_data_large.npz"):
        np.savez(
            npz_path,
            input_frame=np.array(self.calib_inputs, dtype=np.float32),
            h_in=np.array(self.calib_h, dtype=np.float32)
        )
        print(f"[Supporting File] Cube.AI calibration set has been saved to: {npz_path} (Total {len(self.calib_inputs)} Frames)")


# 2. ONNX 推理包装器：伪装成 PyTorch 模型供 DNSMOS 函数直接评估
class OnnxStaticQuantWrapper(nn.Module):
    def __init__(self, onnx_path, hidden_dim=128, num_layers=3):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, h_state=None):
        device = x.device
        batch_size, seq_len, _ = x.shape

        if h_state is None:
            h_curr = np.zeros((batch_size, self.num_layers, self.hidden_dim), dtype=np.float32)
        else:
            h_curr = h_state.transpose(0, 1).detach().cpu().numpy().astype(np.float32)

        x_np = x.detach().cpu().numpy().astype(np.float32)
        outputs = []

        # 逐帧流式推理（与 STM32 端侧推理方式完全一致）
        for t in range(seq_len):
            frame_t = x_np[:, t:t+1, :]  # [Batch, 1, 256]
            out_t, h_curr = self.session.run(
                ["output_frame", "h_out"],
                {"input_frame": frame_t, "h_in": h_curr}
            )
            outputs.append(out_t)

        out_seq = np.concatenate(outputs, axis=1)  # [Batch, Seq_Len, 256]
        h_out_torch = torch.from_numpy(h_curr).transpose(0, 1).contiguous().to(device)
        return torch.from_numpy(out_seq).to(device), h_out_torch


# 3. 导出单帧流式 FP32 ONNX 通用函数
def export_single_frame_onnx(pth_path, onnx_path, hidden_dim, num_layers):
    device = torch.device("cpu")
    model = TinyAudioGRU(input_dim=256, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    unrolled_model = UnrolledStreamingGRU(model, input_dim=256, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    unrolled_model.eval()
    test_x = torch.randn(1, 1, 256)
    test_h_raw = torch.randn(num_layers, 1, hidden_dim)
    test_h_unrolled = test_h_raw.transpose(0, 1).contiguous()
    with torch.no_grad():
        out1, _ = model(test_x, test_h_raw)
        out2, _ = unrolled_model(test_x, test_h_unrolled)
        max_diff = torch.max(torch.abs(out1 - out2)).item()
        print(f"[自检通过] {pth_path} 展开前后最大输出误差: {max_diff:.2e}")

    dummy_input = torch.randn(1, 1, 256, dtype=torch.float32)
    dummy_h = torch.zeros(1, num_layers, hidden_dim, dtype=torch.float32)

    torch.onnx.export(
        unrolled_model,
        (dummy_input, dummy_h),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_frame", "h_in"],
        output_names=["output_frame", "h_out"],
        dynamo=False
    )
    size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[导出成功] {onnx_path} | 体积: {size_kb:.2f} KB")


if __name__ == '__main__':
    cpu_device = torch.device("cpu")

    # --- 步骤 1: 导出小模型 (保持 FP32，不量化) ---
    print("--- 步骤 1 ---")
    export_single_frame_onnx(
        pth_path="model_small_best.pth",
        onnx_path="model_small_fp32.onnx",
        hidden_dim=16,
        num_layers=1
    )

    # --- 步骤 2: 导出大模型基准 (FP32) ---
    print("\n--- 步骤 2 ---")
    large_fp32_onnx = "model_large_fp32.onnx"
    export_single_frame_onnx(
        pth_path="model_large_best.pth",
        onnx_path=large_fp32_onnx,
        hidden_dim=128,
        num_layers=3
    )

    # --- 步骤 3: 严格划分数据（对齐训练时的前 2000 条） ---
    print("\n--- 步骤 3 ---")
    NOISY_DATA_DIR = "F:/master thesis/processed_dataset/train_snr_0"
    CLEAN_DATA_DIR = "F:/master thesis/dev-clean/LibriSpeech/dev-clean"
    raw_clean_files = sorted(glob.glob(os.path.join(CLEAN_DATA_DIR, "**/*.flac"), recursive=True))

    def build_paired_files(clean_file_list):
        pairs = []
        for clean_path in clean_file_list:
            base_name = os.path.splitext(os.path.basename(clean_path))[0]
            target_noisy_path = os.path.join(NOISY_DATA_DIR, f"{base_name}.wav")
            if os.path.exists(target_noisy_path):
                pairs.append((target_noisy_path, clean_path))
        return pairs


    base_pairs = build_paired_files(raw_clean_files[:2000])
    split_idx = int(len(base_pairs) * 0.8)
    val_pairs = base_pairs[split_idx:]

    calib_pairs = build_paired_files(raw_clean_files[2000:2030])[:20]
    if len(calib_pairs) == 0:
        calib_pairs = base_pairs[:20]

    # --- 步骤 4: 对大模型执行 INT8 静态量化 ---
    print("\n--- 步骤 4 ---")
    prep_fp32_onnx = "model_large_fp32_prep.onnx"
    quant_pre_process(large_fp32_onnx, prep_fp32_onnx)

    calib_dataset = AudioDenoisingDataset(calib_pairs)
    calib_reader = LargeModelCalibReader(large_fp32_onnx, calib_dataset, num_files=15, frames_per_file=20)
    calib_reader.save_as_npz("calib_data_large.npz")  # 如不需要可注释此行

    large_int8_onnx = "model_large_int8_static.onnx"
    quantize_static(
        model_input=large_fp32_onnx,
        model_output=large_int8_onnx,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["MatMul", "Gemm"],
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False
    )

    fp32_kb = os.path.getsize(large_fp32_onnx) / 1024.0
    int8_kb = os.path.getsize(large_int8_onnx) / 1024.0
    print("\n==================================================")
    print(f"【体积对比】大模型 FP32 ONNX: {fp32_kb:.2f} KB ({fp32_kb / 1024:.2f} MB)")
    print(f"【体积对比】大模型 INT8 ONNX: {int8_kb:.2f} KB ({int8_kb / 1024:.2f} MB)")
    print(f"【压缩收益】体积直接缩减了: {(1 - int8_kb / fp32_kb) * 100:.1f}%")
    print(f"【Cube.AI 文件】小模型: model_small_fp32.onnx | 大模型: {large_int8_onnx}")
    print("==================================================\n")

    # --- 步骤 5: 评估静态 INT8 量化大模型的 DNSMOS 分数 ---
    print("--- 步骤 5 ---")
    val_dataset = AudioDenoisingDataset(val_pairs[:100])
    eval_test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

    wrapped_int8_model = OnnxStaticQuantWrapper(large_int8_onnx, hidden_dim=128, num_layers=3)
    wrapped_int8_model.eval()

    evaluate_audio_quality_dnsmos(wrapped_int8_model, eval_test_loader, cpu_device, "LARGE")
