"""
PFB v2 - 使用基于窗函数的完美重建实现

这个版本更接近于高质量STFT，使用优化的Kaiser窗实现完美重建。
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import get_window
from typing import Tuple
import sys
from pathlib import Path
import io

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class PFBAnalyzerV2:
    """PFB分析器 v2 - 使用优化的Kaiser窗"""
    
    def __init__(
        self,
        fft_size: int = 128,
        hop_size: int = 64,
        sample_rate: float = 16000,
        kaiser_beta: float = 12.0
    ):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        
        # 创建Kaiser窗
        self.window = get_window(('kaiser', kaiser_beta), fft_size)
        
        # 计算COLA条件补偿（确保完美重建）
        # sum(w^2[n - m*R]) should be constant
        self.cola_compensation = self._compute_cola_compensation()
        
        self.freqs = fftfreq(fft_size, d=1/sample_rate)[:fft_size//2]
        
        print(f"✅ PFB分析器v2初始化完成")
        print(f"   FFT大小: {fft_size}")
        print(f"   跳跃长度: {hop_size}")
        print(f"   Kaiser beta: {kaiser_beta}")
    
    def _compute_cola_compensation(self):
        """计算COLA补偿系数"""
        # 模拟叠加后的窗函数平方和
        num_frames_overlap = self.fft_size // self.hop_size
        window_sum = np.zeros(self.fft_size)
        
        for i in range(num_frames_overlap):
            shifted = np.roll(self.window**2, i * self.hop_size)
            window_sum += shifted
        
        # 避免除以零
        compensation = 1.0 / (window_sum + 1e-12)
        return compensation
    
    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理信号"""
        # 分帧
        num_frames = (len(signal) - self.fft_size) // self.hop_size + 1
        frames = np.zeros((num_frames, self.fft_size))
        
        for i in range(num_frames):
            start = i * self.hop_size
            frames[i] = signal[start:start + self.fft_size]
        
        # 应用窗函数
        windowed = frames * self.window
        
        # FFT
        spectra = fft(windowed, axis=1)
        
        # 只取正频率
        spectra = spectra[:, :self.fft_size//2]
        
        magnitude = np.abs(spectra)
        phase = np.angle(spectra)
        
        return magnitude, phase


class PFBSynthesizerV2:
    """PFB合成器 v2"""
    
    def __init__(
        self,
        fft_size: int = 128,
        hop_size: int = 64,
        sample_rate: float = 16000,
        kaiser_beta: float = 12.0
    ):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        
        # 创建Kaiser窗（与分析器相同）
        self.window = get_window(('kaiser', kaiser_beta), fft_size)
        
        # 计算COLA补偿
        num_frames_overlap = self.fft_size // self.hop_size
        window_sum = np.zeros(self.fft_size)
        for i in range(num_frames_overlap):
            shifted = np.roll(self.window**2, i * self.hop_size)
            window_sum += shifted
        self.cola_compensation = 1.0 / (window_sum + 1e-12)
        
        print(f"✅ PFB合成器v2初始化完成")
    
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
        
        # 应用窗函数
        windowed = time_domain * self.window
        
        # 应用COLA补偿
        windowed *= self.cola_compensation
        
        # Overlap-Add
        output_length = (num_frames - 1) * self.hop_size + self.fft_size
        output = np.zeros(output_length)
        
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.fft_size
            output[start:end] += windowed[i]
        
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("PFB v2 测试")
    print("=" * 60)
    
    # 创建分析器和合成器
    analyzer = PFBAnalyzerV2(
        fft_size=128,
        hop_size=64,
        sample_rate=16000,
        kaiser_beta=12.0
    )
    
    synthesizer = PFBSynthesizerV2(
        fft_size=128,
        hop_size=64,
        sample_rate=16000,
        kaiser_beta=12.0
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
        print(f"   ⚠️  SNR较低")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
