"""
PFB (Polyphase Filter Bank) 合成器实现

基于 C 模型的严格实现，与 dios_ssp_share_subband.c 逻辑一致。
已验证：使用 -648 采样群延时补偿可实现正确重建。
"""

import numpy as np
from scipy.fft import irfft
from typing import Optional
from pathlib import Path
import sys

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))


class PFBSynthesis:
    """PFB合成器 - 基于C模型的严格实现"""

    def __init__(
        self,
        fft_len: int = 256,
        win_len: int = 768,
        frm_len: int = 128,
        filter_coef: Optional[np.ndarray] = None,
        preload_filter: Optional[str] = None,
        scale: float = -256.0
    ):
        """
        初始化PFB合成器（严格遵循C模型逻辑）

        Args:
            fft_len: FFT长度
            win_len: 窗长（滤波器长度）
            frm_len: 帧长（跳跃长度）
            filter_coef: 滤波器系数（可选）
            preload_filter: 预加载滤波器文件路径
            scale: 重建缩放因子（默认 -256.0）
        """
        self.fft_len = fft_len
        self.win_len = win_len
        self.frm_len = frm_len
        self.ppf_tap = win_len // fft_len  # 3
        self.scale = scale

        # 加载滤波器系数
        if preload_filter:
            self.lpf_coef = np.load(preload_filter)
            print(f"[INFO] 从文件加载滤波器: {preload_filter}")
        elif filter_coef is not None:
            self.lpf_coef = np.array(filter_coef, dtype=np.float32)
            assert len(self.lpf_coef) == win_len
        else:
            # 尝试从 assets 目录加载
            assets_dir = Path(__file__).parent.parent / "assets"
            coef_path = assets_dir / "pfb_filter_coef_768.npy"

            if coef_path.exists():
                self.lpf_coef = np.load(str(coef_path))
                print(f"[INFO] Loaded C model filter coefficients")
            else:
                from scipy.signal import firwin
                self.lpf_coef = firwin(win_len, 0.5, window='hann', scale=False).astype(np.float32)
                print(f"[WARNING] Using designed filter")

        # 缓冲区初始化
        self.comp_in = np.zeros(fft_len, dtype=np.float32)
        self.comp_out = np.zeros(win_len, dtype=np.float32)

        # 群延时补偿（已知为 -648 采样 @ 16kHz）
        self.group_delay = -648

        print(f"[INFO] PFB Synthesizer C Model initialized:")
        print(f"   FFT_LEN = {fft_len}, WIN_LEN = {win_len}, FRM_LEN = {frm_len}")
        print(f"   Scale = {self.scale}")
        print(f"   群延时补偿 = {self.group_delay} 采样")

    def reset(self):
        """重置状态"""
        self.comp_in[:] = 0.0
        self.comp_out[:] = 0.0

    def synthesize(self, in_buf: np.ndarray) -> np.ndarray:
        """
        合成频域复数为时域信号

        Args:
            in_buf: 输入复数频谱 (subband_num,) 其中 subband_num = fft_len // 2 + 1

        Returns:
            out_buf: 输出时域帧 (frm_len,)
        """
        subband_num = self.fft_len // 2 + 1
        assert len(in_buf) == subband_num

        # 步骤1: 复数转RFFT输入格式
        fftin_complex = in_buf.copy()

        # 步骤2: IFFT (C代码第272行)
        fftout = irfft(fftin_complex, n=self.fft_len)

        # 步骤3: 复制到comp_in (C代码第274-277行)
        for i in range(self.fft_len):
            self.comp_in[i] = fftout[i]

        # 步骤4: 多相合成 (C代码第278-285行) - 累积到comp_out
        for i in range(self.ppf_tap):
            for j in range(self.fft_len):
                k = i * self.fft_len + j
                self.comp_out[k] += self.lpf_coef[k] * self.comp_in[self.fft_len - j - 1]

        # 步骤5: 输出当前帧 (C代码第287-290行)
        out_buf = np.zeros(self.frm_len, dtype=np.float32)
        for i in range(self.frm_len):
            out_buf[i] = self.comp_out[i] * self.frm_len * self.scale

        # 步骤6: 移位为下一帧准备 (C代码第292-300行)
        for i in range(self.win_len - self.frm_len):
            self.comp_out[i] = self.comp_out[i + self.frm_len]
        for i in range(self.win_len - self.frm_len, self.win_len):
            self.comp_out[i] = 0.0

        return out_buf

    def process(self, spectrum: np.ndarray) -> np.ndarray:
        """
        处理整个频谱序列

        Args:
            spectrum: 复数频谱序列 (num_frames, subband_num)

        Returns:
            重构的时域信号 (samples,)
        """
        # 重置状态
        self.reset()

        # 计算帧数
        num_frames = spectrum.shape[0]

        # 预分配输出
        output_samples = []

        # 逐帧处理
        for frame_idx in range(num_frames):
            output_frame = self.synthesize(spectrum[frame_idx])
            output_samples.append(output_frame)

        # 拼接输出
        output = np.concatenate(output_samples)

        # 应用群延时补偿（-648 采样）
        output = np.roll(output, self.group_delay)
        output = output[:len(output) + self.group_delay]

        return output

    def process_complex(self, spectrum: np.ndarray) -> np.ndarray:
        """
        直接处理复数频谱（别名）
        """
        return self.process(spectrum)


if __name__ == "__main__":
    print("=" * 60)
    print("PFB合成器测试 (C Model - 完美重建)")
    print("=" * 60)

    # 从 analysis 导入
    from pfb_analysis import PFBAnalysis

    # 创建分析器和合成器
    analyzer = PFBAnalysis(fft_len=256, win_len=768, frm_len=128)
    synthesizer = PFBSynthesis(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

    # 生成测试信号
    fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    test_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 1000 * t) +
        0.2 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)

    print(f"\n测试信号: {len(test_signal)} 采样, {duration} 秒")

    # 分析
    print("\n[分析] PFB分析...")
    complex_spectrum = analyzer.process(test_signal)
    print(f"   输出频谱: {complex_spectrum.shape[0]} 帧 x {complex_spectrum.shape[1]} 子带")

    # 合成
    print("\n[合成] PFB合成...")
    reconstructed = synthesizer.process(complex_spectrum)
    print(f"   重构信号: {len(reconstructed)} 采样")

    # 对齐长度
    min_len = min(len(test_signal), len(reconstructed))
    test_signal_aligned = test_signal[:min_len]
    reconstructed_aligned = reconstructed[:min_len]

    # 计算重建质量
    error = test_signal_aligned - reconstructed_aligned
    mse = np.mean(error**2)
    max_error = np.max(np.abs(error))
    signal_power = np.mean(test_signal_aligned**2)
    snr_db = 10 * np.log10(signal_power / (mse + 1e-12))

    # 计算相关系数
    correlation = np.corrcoef(test_signal_aligned, reconstructed_aligned)[0, 1]

    print(f"\n[验证] 完美重建验证:")
    print(f"   MSE: {mse:.2e}")
    print(f"   最大绝对误差: {max_error:.2e}")
    print(f"   SNR: {snr_db:.2f} dB")
    print(f"   相关系数: {correlation:.4f}")

    if snr_db > 40 and correlation > 0.7:
        print(f"   ✅ 完美重建验证通过！")
    else:
        print(f"   ⚠️  重建质量有待提高")

    print("\n" + "=" * 60)
    print("✅ PFB合成器测试完成")
    print("=" * 60)
