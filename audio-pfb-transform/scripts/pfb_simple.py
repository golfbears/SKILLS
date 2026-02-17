"""
简化的PFB实现 - 使用矩阵运算
"""

import numpy as np
from scipy.fft import fft, ifft
from typing import Tuple
import sys
from pathlib import Path

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from filter_design import compute_kaiser_filter, compute_polyphase_filters


class PFBAnalyzerSimple:
    """简化的PFB分析器"""
    
    def __init__(
        self,
        fft_size: int = 128,
        hop_size: int = 64,
        filter_length: int = 768,
        sample_rate: float = 16000,
        kaiser_beta: float = 12.0
    ):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        
        # 计算滤波器
        self.h = compute_kaiser_filter(
            fft_size=fft_size,
            filter_length=filter_length,
            sample_rate=sample_rate,
            kaiser_beta=kaiser_beta,
            cutoff_ratio=0.5
        )
        
        # 计算多相滤波器
        self.polyphase_filters = compute_polyphase_filters(self.h, fft_size)
        
        print(f"✅ PFB分析器初始化完成")
    
    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理信号"""
        # 填充信号
        pad_length = self.filter_length - self.hop_size
        signal_padded = np.pad(signal, (pad_length, 0))
        
        # 分帧
        num_frames = (len(signal_padded) - self.filter_length) // self.hop_size + 1
        frames = np.zeros((num_frames, self.filter_length))
        
        for i in range(num_frames):
            start = i * self.hop_size
            frames[i] = signal_padded[start:start + self.filter_length]
        
        # 多相滤波
        # 重塑为 (num_frames, fft_size, filter_length//fft_size)
        frames_reshaped = frames.reshape(num_frames, self.fft_size, -1)
        
        # 应用多相滤波器 (逐帧广播)
        filtered = np.zeros((num_frames, self.fft_size), dtype=np.complex128)
        for i in range(self.fft_size):
            # 卷积: sum(frames[:, i, :] * filters[i, :])
            filtered[:, i] = np.sum(frames_reshaped[:, i, :] * self.polyphase_filters[i], axis=1)
        
        # FFT
        spectra = fft(filtered, axis=1)
        
        # 只取正频率
        spectra = spectra[:, :self.fft_size//2]
        
        magnitude = np.abs(spectra)
        phase = np.angle(spectra)
        
        return magnitude, phase


class PFBSynthesizerSimple:
    """简化的PFB合成器"""
    
    def __init__(
        self,
        fft_size: int = 128,
        hop_size: int = 64,
        filter_length: int = 768,
        sample_rate: float = 16000,
        kaiser_beta: float = 12.0
    ):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        
        # 计算滤波器（与分析器相同）
        self.h = compute_kaiser_filter(
            fft_size=fft_size,
            filter_length=filter_length,
            sample_rate=sample_rate,
            kaiser_beta=kaiser_beta,
            cutoff_ratio=0.5
        )
        
        # 计算多相滤波器
        self.polyphase_filters = compute_polyphase_filters(self.h, fft_size)
        
        self.delay = (filter_length - 1) // 2
        
        print(f"✅ PFB合成器初始化完成")
    
    def process(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """重构信号"""
        # 重构完整频谱
        num_frames = magnitude.shape[0]
        spectra = np.zeros((num_frames, self.fft_size), dtype=np.complex128)
        spectra[:, :self.fft_size//2] = magnitude * np.exp(1j * phase)
        spectra[:, self.fft_size//2:] = np.conj(spectra[:, :self.fft_size//2][:, ::-1])
        
        # IFFT
        time_domain = ifft(spectra, axis=1).real
        
        # 多相合成
        polyphase_len = self.filter_length // self.fft_size
        synthesized = np.zeros((num_frames, self.filter_length))
        
        for i in range(self.fft_size):
            for j in range(polyphase_len):
                idx = i + j * self.fft_size
                synthesized[:, idx] += time_domain[:, i] * self.polyphase_filters[i, j]
        
        # Overlap-Add
        output_length = num_frames * self.hop_size
        output = np.zeros(output_length + self.filter_length)
        
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.filter_length
            output[start:end] += synthesized[i]
        
        # 截取有效输出
        output = output[:output_length]
        
        # 应用时延补偿
        output = np.roll(output, self.delay)
        output = output[:output_length - self.delay]
        
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("简化PFB测试")
    print("=" * 60)
    
    # 创建分析器和合成器
    analyzer = PFBAnalyzerSimple(
        fft_size=128,
        hop_size=64,
        filter_length=768,
        sample_rate=16000
    )
    
    synthesizer = PFBSynthesizerSimple(
        fft_size=128,
        hop_size=64,
        filter_length=768,
        sample_rate=16000
    )
    
    # 生成测试信号
    duration = 2.0
    t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
    test_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 1000 * t) +
        0.2 * np.sin(2 * np.pi * 3000 * t) +
        0.1 * np.random.randn(len(t))
    )
    
    print(f"\n📊 测试信号: {len(test_signal)} 采样")
    
    # 分析
    print("\n🔄 分析...")
    magnitude, phase = analyzer.process(test_signal)
    print(f"   输出频谱: {magnitude.shape[0]} 帧 x {magnitude.shape[1]} 频率bins")
    
    # 合成
    print("\n🔄 合成...")
    reconstructed = synthesizer.process(magnitude, phase)
    print(f"   重构信号: {len(reconstructed)} 采样")
    
    # 验证
    min_len = min(len(test_signal), len(reconstructed))
    test_aligned = test_signal[:min_len]
    reconstructed_aligned = reconstructed[:min_len]
    
    mse = np.mean((test_aligned - reconstructed_aligned)**2)
    max_error = np.max(np.abs(test_aligned - reconstructed_aligned))
    snr = 10 * np.log10(np.mean(test_aligned**2) / (mse + 1e-12))
    
    print(f"\n📊 完美重建验证:")
    print(f"   MSE: {mse:.2e}")
    print(f"   最大误差: {max_error:.2e}")
    print(f"   SNR: {snr:.2f} dB")
    
    if snr > 60:
        print(f"   ✅ 完美重建验证通过！")
    else:
        print(f"   ⚠️  SNR较低，需要进一步优化")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
