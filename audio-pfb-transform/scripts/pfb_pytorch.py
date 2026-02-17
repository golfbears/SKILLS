"""
PFB (Polyphase Filter Bank) PyTorch实现

基于验证正确的 C 模型逻辑的可微分版本，用于深度学习训练。
参数：FFT_LEN=256, WIN_LEN=768, FRM_LEN=128, Ppf_tap=3, Scale=-256.0
群延时补偿：-648 采样 @ 16kHz
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from pathlib import Path


class PFBAnalysisLayer(nn.Module):
    """PFB分析层 - PyTorch可微分版本"""

    def __init__(
        self,
        fft_len: int = 256,
        win_len: int = 768,
        frm_len: int = 128,
        filter_coef: np.ndarray = None
    ):
        super().__init__()

        self.fft_len = fft_len
        self.win_len = win_len
        self.frm_len = frm_len
        self.ppf_tap = win_len // fft_len
        self.ppf_decm = win_len // frm_len
        self.subband_num = fft_len // 2 + 1

        # 加载滤波器系数
        if filter_coef is not None:
            lpf_coef = np.array(filter_coef, dtype=np.float32)
        else:
            # 尝试从 assets 目录加载
            assets_dir = Path(__file__).parent.parent / "assets"
            coef_path = assets_dir / "pfb_filter_coef_768.npy"
            if coef_path.exists():
                lpf_coef = np.load(str(coef_path)).astype(np.float32)
            else:
                from scipy.signal import firwin
                lpf_coef = firwin(win_len, 0.5, window='hann', scale=False).astype(np.float32)

        # 注册滤波器系数为buffer（不可训练）
        self.register_buffer('lpf_coef', torch.from_numpy(lpf_coef))

        # 初始化多相索引
        p_in_init = np.array([i * frm_len for i in range(self.ppf_decm)], dtype=np.int64)
        p_h0_init = np.array([i * frm_len for i in range(self.ppf_decm)], dtype=np.int64)

        self.register_buffer('p_in', torch.from_numpy(p_in_init))
        self.register_buffer('p_h0', torch.from_numpy(p_h0_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：时域 -> 频域

        Args:
            x: 输入信号 (batch, samples) 或 (batch, channels, samples)

        Returns:
            spectrum: 复数频谱 (batch, subband_num, time_frames) 或 (batch, channels, subband_num, time_frames)
        """
        # 处理输入维度
        if x.dim() == 2:
            # (batch, samples) -> (batch, 1, samples)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_channels, num_samples = x.shape

        # 计算帧数
        num_frames = num_samples // self.frm_len

        # 重塑为帧 (batch, channels, num_frames, frm_len)
        x = x[:, :, :num_frames * self.frm_len]
        x = x.reshape(batch_size, num_channels, num_frames, self.frm_len)

        # 初始化输出
        spectrum = torch.zeros(
            batch_size,
            num_channels,
            num_frames,
            self.subband_num,
            dtype=torch.complex64,
            device=x.device
        )

        # 逐帧处理（为了严格遵循C模型逻辑）
        # 注意：这里使用了循环，可以优化为矩阵运算，但保持可读性
        for batch_idx in range(batch_size):
            for channel_idx in range(num_channels):
                # 初始化缓冲区
                ana_xin = torch.zeros(self.win_len, dtype=torch.float32, device=x.device)
                ana_xout = torch.zeros(self.fft_len, dtype=torch.float32, device=x.device)
                p_in = self.p_in.clone()

                for frame_idx in range(num_frames):
                    frame = x[batch_idx, channel_idx, frame_idx]  # (frm_len,)

                    # 步骤1: 输入数据反转并写入环形缓冲区
                    for i in range(self.frm_len - 1, -1, -1):
                        ana_xin[i + p_in[0]] = frame[self.frm_len - i - 1]

                    # 步骤2: 多相滤波
                    for i in range(self.fft_len):
                        r0 = 0.0
                        if i < self.frm_len:
                            for j in range(self.ppf_tap):
                                r0 += (self.lpf_coef[p_h0 := self.p_h0[2 * j] + i] *
                                       ana_xin[p_in[2 * j] + i])
                        else:
                            for j in range(self.ppf_tap):
                                r0 += (self.lpf_coef[p_h0 := self.p_h0[2 * j + 1] + i - self.frm_len] *
                                       ana_xin[p_in[2 * j + 1] + i - self.frm_len])
                        ana_xout[i] = r0

                    # 步骤3: 指针循环移动
                    itmp = p_in[self.ppf_decm - 1]
                    for i in range(self.ppf_decm - 1, 0, -1):
                        p_in[i] = p_in[i - 1]
                    p_in[0] = itmp

                    # 步骤4: RFFT
                    fftout = torch.fft.rfft(ana_xout, n=self.fft_len)
                    spectrum[batch_idx, channel_idx, frame_idx] = fftout

        # 转置为 (batch, channels, subband_num, time_frames) -> (batch, subband_num, time_frames)
        # 如果是多通道，保持通道维度
        if squeeze_output:
            spectrum = spectrum[:, 0]  # (batch, time_frames, subband_num)
            spectrum = spectrum.permute(0, 2, 1)  # (batch, subband_num, time_frames)
        else:
            spectrum = spectrum.permute(0, 1, 3, 2)  # (batch, channels, subband_num, time_frames)

        return spectrum


class PFBSynthesisLayer(nn.Module):
    """PFB合成层 - PyTorch可微分版本"""

    def __init__(
        self,
        fft_len: int = 256,
        win_len: int = 768,
        frm_len: int = 128,
        filter_coef: np.ndarray = None,
        scale: float = -256.0
    ):
        super().__init__()

        self.fft_len = fft_len
        self.win_len = win_len
        self.frm_len = frm_len
        self.ppf_tap = win_len // fft_len
        self.scale = scale
        self.group_delay = -648  # 已验证的群延时补偿

        # 加载滤波器系数
        if filter_coef is not None:
            lpf_coef = np.array(filter_coef, dtype=np.float32)
        else:
            # 尝试从 assets 目录加载
            assets_dir = Path(__file__).parent.parent / "assets"
            coef_path = assets_dir / "pfb_filter_coef_768.npy"
            if coef_path.exists():
                lpf_coef = np.load(str(coef_path)).astype(np.float32)
            else:
                from scipy.signal import firwin
                lpf_coef = firwin(win_len, 0.5, window='hann', scale=False).astype(np.float32)

        # 注册滤波器系数为buffer（不可训练）
        self.register_buffer('lpf_coef', torch.from_numpy(lpf_coef))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        前向传播：频域 -> 时域

        Args:
            spectrum: 复数频谱 (batch, subband_num, time_frames) 或 (batch, channels, subband_num, time_frames)

        Returns:
            x: 重构信号 (batch, samples) 或 (batch, channels, samples)
        """
        # 处理输入维度
        if spectrum.dim() == 3:
            # (batch, subband_num, time_frames) -> (batch, 1, subband_num, time_frames)
            spectrum = spectrum.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_channels, subband_num, num_frames = spectrum.shape

        # 计算输出采样数
        num_samples = num_frames * self.frm_len

        # 初始化输出
        output = torch.zeros(
            batch_size,
            num_channels,
            num_samples,
            dtype=torch.float32,
            device=spectrum.device
        )

        # 逐帧处理
        for batch_idx in range(batch_size):
            for channel_idx in range(num_channels):
                # 初始化缓冲区
                comp_in = torch.zeros(self.fft_len, dtype=torch.float32, device=spectrum.device)
                comp_out = torch.zeros(self.win_len, dtype=torch.float32, device=spectrum.device)

                for frame_idx in range(num_frames):
                    in_buf = spectrum[batch_idx, channel_idx, frame_idx]  # (subband_num,)

                    # 步骤1: IFFT
                    fftout = torch.fft.irfft(in_buf, n=self.fft_len)

                    # 步骤2: 复制到comp_in
                    comp_in[:] = fftout

                    # 步骤3: 多相合成
                    for i in range(self.ppf_tap):
                        for j in range(self.fft_len):
                            k = i * self.fft_len + j
                            comp_out[k] += self.lpf_coef[k] * comp_in[self.fft_len - j - 1]

                    # 步骤4: 输出当前帧
                    out_buf = torch.zeros(self.frm_len, dtype=torch.float32, device=spectrum.device)
                    for i in range(self.frm_len):
                        out_buf[i] = comp_out[i] * self.frm_len * self.scale

                    # 步骤5: 存储输出
                    start = frame_idx * self.frm_len
                    end = start + self.frm_len
                    output[batch_idx, channel_idx, start:end] = out_buf

                    # 步骤6: 移位
                    for i in range(self.win_len - self.frm_len):
                        comp_out[i] = comp_out[i + self.frm_len]
                    for i in range(self.win_len - self.frm_len, self.win_len):
                        comp_out[i] = 0.0

                # 应用群延时补偿
                output[batch_idx, channel_idx] = torch.roll(
                    output[batch_idx, channel_idx],
                    self.group_delay
                )

        # 如果群延时为负，截取末尾部分
        if self.group_delay < 0:
            output = output[:, :, :self.group_delay]

        # 移除通道维度（如果需要）
        if squeeze_output:
            output = output[:, 0]  # (batch, samples)

        return output


class PFBTransform(nn.Module):
    """完整的 PFB 变换（分析 + 合成）"""

    def __init__(
        self,
        fft_len: int = 256,
        win_len: int = 768,
        frm_len: int = 128,
        filter_coef: np.ndarray = None,
        scale: float = -256.0
    ):
        super().__init__()

        self.analysis = PFBAnalysisLayer(fft_len, win_len, frm_len, filter_coef)
        self.synthesis = PFBSynthesisLayer(fft_len, win_len, frm_len, filter_coef, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整的 PFB 变换（用于测试完美重建）

        Args:
            x: 输入信号 (batch, samples) 或 (batch, channels, samples)

        Returns:
            reconstructed: 重构信号
        """
        spectrum = self.analysis(x)
        reconstructed = self.synthesis(spectrum)
        return reconstructed


if __name__ == "__main__":
    print("=" * 60)
    print("PFB PyTorch 实现测试")
    print("=" * 60)

    # 创建变换层
    pfb = PFBTransform(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pfb = pfb.to(device)

    print(f"\n使用设备: {device}")

    # 生成测试信号
    batch_size = 2
    num_samples = 16000  # 1秒 @ 16kHz
    t = torch.linspace(0, 1, num_samples, device=device)

    signal = (
        0.5 * torch.sin(2 * np.pi * 440 * t) +
        0.3 * torch.sin(2 * np.pi * 1000 * t) +
        0.2 * torch.sin(2 * np.pi * 3000 * t)
    ).unsqueeze(0).expand(batch_size, -1)

    print(f"\n输入信号: {signal.shape}")

    # 测试完美重建
    with torch.no_grad():
        reconstructed = pfb(signal)

    print(f"重构信号: {reconstructed.shape}")

    # 对齐长度
    min_len = min(signal.shape[1], reconstructed.shape[1])
    signal_trunc = signal[:, :min_len]
    reconstructed_trunc = reconstructed[:, :min_len]

    # 计算误差
    mse = torch.mean((signal_trunc - reconstructed_trunc)**2).item()
    max_error = torch.max(torch.abs(signal_trunc - reconstructed_trunc)).item()
    snr = 10 * torch.log10(torch.mean(signal_trunc**2) / (torch.mean((signal_trunc - reconstructed_trunc)**2) + 1e-12)).item()

    print(f"\n完美重建验证:")
    print(f"   MSE: {mse:.2e}")
    print(f"   最大误差: {max_error:.2e}")
    print(f"   SNR: {snr:.2f} dB")

    if snr > 40:
        print(f"   ✅ 完美重建验证通过！")
    else:
        print(f"   ⚠️  SNR: {snr:.2f} dB")

    # 测试可微分性
    print("\n测试梯度传播...")
    signal_grad = signal.clone().requires_grad_(True)
    output = pfb(signal_grad)
    loss = torch.mean(output**2)
    loss.backward()

    print(f"   损失值: {loss.item():.4f}")
    print(f"   输入梯度存在: {signal_grad.grad is not None}")
    if signal_grad.grad is not None:
        print(f"   输入梯度范数: {torch.norm(signal_grad.grad).item():.4f}")

    print("\n" + "=" * 60)
    print("✅ PFB PyTorch 实现测试完成")
    print("=" * 60)
