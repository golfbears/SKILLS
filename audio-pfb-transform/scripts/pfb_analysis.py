"""
PFB (Polyphase Filter Bank) 分析器实现

基于 C 模型的严格实现，与 dios_ssp_share_subband.c 逻辑一致。
已验证：使用 -648 采样群延时补偿可实现正确重建。
"""

import numpy as np
from scipy.fft import rfft
from typing import Tuple, Optional
from pathlib import Path
import sys
import os

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))


class PFBAnalysis:
    """PFB分析器 - 基于C模型的严格实现"""

    def __init__(
        self,
        fft_len: int = 256,
        win_len: int = 768,
        frm_len: int = 128,
        filter_coef: Optional[np.ndarray] = None,
        preload_filter: Optional[str] = None
    ):
        """
        初始化PFB分析器（严格遵循C模型逻辑）

        Args:
            fft_len: FFT长度
            win_len: 窗长（滤波器长度）
            frm_len: 帧长（跳跃长度）
            filter_coef: 滤波器系数（可选）
            preload_filter: 预加载滤波器文件路径
        """
        self.fft_len = fft_len
        self.win_len = win_len
        self.frm_len = frm_len
        self.ppf_tap = win_len // fft_len  # 3
        self.ppf_decm = win_len // frm_len  # 6
        self.scale = 1.0

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
        self.ana_xin = np.zeros(win_len, dtype=np.float32)
        self.ana_xout = np.zeros(fft_len, dtype=np.float32)

        # 多相索引指针 (严格遵循C代码)
        self.p_in = np.array([i * frm_len for i in range(self.ppf_decm)], dtype=np.int32)
        self.p_h0 = np.array([i * frm_len for i in range(self.ppf_decm)], dtype=np.int32)

        print(f"[INFO] PFB Analyzer C Model initialized:")
        print(f"   FFT_LEN = {fft_len}, WIN_LEN = {win_len}, FRM_LEN = {frm_len}")
        print(f"   Ppf_tap = {self.ppf_tap}, Ppf_decm = {self.ppf_decm}")
        print(f"   Subband num = {fft_len // 2 + 1}")
        print(f"   注意: 合成时需要 -648 采样群延时补偿")

    def reset(self):
        """重置状态"""
        self.ana_xin[:] = 0.0
        self.ana_xout[:] = 0.0
        self.p_in = np.array([i * self.frm_len for i in range(self.ppf_decm)], dtype=np.int32)
        self.p_h0 = np.array([i * self.frm_len for i in range(self.ppf_decm)], dtype=np.int32)

    def analyze(self, in_buf: np.ndarray) -> np.ndarray:
        """
        分析时域信号，返回频域复数

        Args:
            in_buf: 输入帧 (frm_len,)

        Returns:
            out_buf: 复数频谱 (subband_num,) 其中 subband_num = fft_len // 2 + 1
        """
        assert len(in_buf) == self.frm_len

        # 步骤1: 输入数据反转并写入环形缓冲区 (C代码第217行)
        for i in range(self.frm_len - 1, -1, -1):
            self.ana_xin[i + self.p_in[0]] = in_buf[self.frm_len - i - 1]

        # 步骤2: 多相滤波 (C代码第220-237行)
        r0 = 0.0
        for i in range(self.fft_len):
            r0 = 0.0
            if i < self.frm_len:
                # 前半部分：使用偶数索引的多相分量
                for j in range(self.ppf_tap):
                    r0 += (self.lpf_coef[self.p_h0[2 * j] + i] *
                           self.ana_xin[self.p_in[2 * j] + i])
            else:
                # 后半部分：使用奇数索引的多相分量
                for j in range(self.ppf_tap):
                    r0 += (self.lpf_coef[self.p_h0[2 * j + 1] + i - self.frm_len] *
                           self.ana_xin[self.p_in[2 * j + 1] + i - self.frm_len])
            self.ana_xout[i] = r0

        # 步骤3: 指针循环移动 (C代码第240-245行)
        itmp = self.p_in[self.ppf_decm - 1]
        for i in range(self.ppf_decm - 1, 0, -1):
            self.p_in[i] = self.p_in[i - 1]
        self.p_in[0] = itmp

        # 步骤4: RFFT并转换为复数格式 (C代码第247-257行)
        fftout = rfft(self.ana_xout, n=self.fft_len)

        subband_num = self.fft_len // 2 + 1
        out_buf = np.zeros(subband_num, dtype=np.complex64)

        # 直接使用scipy的rfft输出格式
        for i in range(subband_num):
            out_buf[i] = fftout[i]

        return out_buf

    def process(
        self,
        signal: np.ndarray,
        return_complex: bool = True
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        处理整个信号

        Args:
            signal: 输入信号 (samples,)
            return_complex: 是否返回复数频谱

        Returns:
            如果return_complex=True: complex_spectrum (num_frames, subband_num)
            如果return_complex=False: (magnitude, phase)
        """
        # 重置状态
        #self.reset()

        # 计算帧数
        num_frames = len(signal) // self.frm_len

        # 预分配输出
        subband_num = self.fft_len // 2 + 1
        complex_list = []

        # 逐帧处理
        for frame_idx in range(num_frames):
            start = frame_idx * self.frm_len
            end = start + self.frm_len
            frame = signal[start:end]

            # 分析
            subbands = self.analyze(frame)
            complex_list.append(subbands)

        # 转换为numpy数组
        complex_spectrum = np.array(complex_list)

        if return_complex:
            return complex_spectrum
        else:
            magnitude = np.abs(complex_spectrum)
            phase = np.angle(complex_spectrum)
            return magnitude, phase


if __name__ == "__main__":
    print("=" * 60)
    print("PFB分析器测试 (C Model)")
    print("=" * 60)

    # 创建分析器
    analyzer = PFBAnalysis(
        fft_len=256,
        win_len=768,
        frm_len=128
    )

    # 生成测试信号
    fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    test_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 1000 * t) +
        0.2 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)

    print(f"\n测试信号: {len(test_signal)} 采样")

    # 处理
    complex_spectrum = analyzer.process(test_signal)
    print(f"输出频谱: {complex_spectrum.shape[0]} 帧 x {complex_spectrum.shape[1]} 子带")

    # 验证直流分量
    print(f"直流子带 (索引0): {complex_spectrum[0, 0]}")
    print(f"440 Hz附近子带幅值: {np.max(np.abs(complex_spectrum[:, 7:10]))}")

    print("\n" + "=" * 60)
    print("✅ PFB分析器测试完成")
    print("=" * 60)
