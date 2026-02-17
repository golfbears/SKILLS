"""
PFB与STFT对比测试脚本

全面对比PFB和传统STFT在各种场景下的性能和效果。
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import signal
import time
from typing import Tuple, Dict
from pathlib import Path
import sys

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入PFB模块
from pfb_analysis import PFBAnalysis, PFBAnalysisBatch
from pfb_synthesis import PFBSynthesis, PFBSynthesisBatch
from pfb_pytorch import PFBTransformLayer
from filter_design import KiserFilterDesigner


def compute_stft(
    signal: np.ndarray,
    fft_size: int,
    hop_size: int,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """计算STFT"""
    frequencies, times, Zxx = signal.stft(
        signal,
        nperseg=fft_size,
        noverlap=fft_size - hop_size,
        window=window
    )
    
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    return magnitude, phase


def compute_istft(
    magnitude: np.ndarray,
    phase: np.ndarray,
    hop_size: int
) -> np.ndarray:
    """计算ISTFT"""
    Zxx = magnitude * np.exp(1j * phase)
    _, reconstructed = signal.istft(Zxx, noverlap=magnitude.shape[0] - hop_size)
    return reconstructed


class PFBvsSTFTComparer:
    """PFB与STFT对比器"""
    
    def __init__(
        self,
        fft_size: int = 128,
        hop_size: int = 64,
        filter_length: int = 768,
        sample_rate: float = 16000
    ):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        
        # 创建PFB分析器和合成器
        self.pfb_analysis = PFBAnalysis(
            fft_size=fft_size,
            hop_size=hop_size,
            filter_length=filter_length,
            sample_rate=sample_rate
        )
        
        self.pfb_synthesis = PFBSynthesis(
            fft_size=fft_size,
            hop_size=hop_size,
            filter_length=filter_length,
            sample_rate=sample_rate
        )
        
        # PyTorch版本
        self.pfb_pytorch = PFBTransformLayer(
            fft_size=fft_size,
            hop_size=hop_size,
            filter_length=filter_length,
            sample_rate=sample_rate
        )
        
        if torch.cuda.is_available():
            self.pfb_pytorch = self.pfb_pytorch.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        print(f"✅ 对比器初始化完成（设备: {self.device}）")
    
    def perfect_reconstruction_test(self, signal: np.ndarray) -> Dict:
        """完美重建测试"""
        print("\n" + "=" * 60)
        print("📊 完美重建测试")
        print("=" * 60)
        
        results = {}
        
        # PFB
        print("\n🔄 PFB处理...")
        pfb_mag, pfb_ph = self.pfb_analysis.process(signal)
        pfb_reconstructed = self.pfb_synthesis.process(pfb_mag, pfb_ph)
        
        # 对齐
        min_len = min(len(signal), len(pfb_reconstructed))
        signal_aligned = signal[:min_len]
        pfb_aligned = pfb_reconstructed[:min_len]
        
        pfb_error = signal_aligned - pfb_aligned
        pfb_mse = np.mean(pfb_error**2)
        pfb_snr = 10 * np.log10(np.mean(signal_aligned**2) / (pfb_mse + 1e-12))
        
        results['pfb'] = {
            'mse': pfb_mse,
            'snr': pfb_snr,
            'max_error': np.max(np.abs(pfb_error))
        }
        
        print(f"   MSE: {pfb_mse:.2e}")
        print(f"   SNR: {pfb_snr:.2f} dB")
        print(f"   最大误差: {results['pfb']['max_error']:.2e}")
        
        # STFT
        print("\n🔄 STFT处理...")
        stft_mag, stft_ph = compute_stft(signal, self.fft_size, self.hop_size)
        stft_reconstructed = compute_istft(stft_mag, stft_ph, self.hop_size)
        
        # 对齐
        min_len_stft = min(len(signal), len(stft_reconstructed))
        signal_aligned_stft = signal[:min_len_stft]
        stft_aligned = stft_reconstructed[:min_len_stft]
        
        stft_error = signal_aligned_stft - stft_aligned
        stft_mse = np.mean(stft_error**2)
        stft_snr = 10 * np.log10(np.mean(signal_aligned_stft**2) / (stft_mse + 1e-12))
        
        results['stft'] = {
            'mse': stft_mse,
            'snr': stft_snr,
            'max_error': np.max(np.abs(stft_error))
        }
        
        print(f"   MSE: {stft_mse:.2e}")
        print(f"   SNR: {stft_snr:.2f} dB")
        print(f"   最大误差: {results['stft']['max_error']:.2e}")
        
        # 对比
        print(f"\n📈 对比:")
        print(f"   PFB SNR: {pfb_snr:.2f} dB")
        print(f"   STFT SNR: {stft_snr:.2f} dB")
        print(f"   差异: {pfb_snr - stft_snr:.2f} dB")
        
        if pfb_snr > stft_snr:
            print(f"   ✅ PFB重建质量更优")
        else:
            print(f"   ⚠️  STFT重建质量更优")
        
        return results
    
    def frequency_response_test(self) -> Dict:
        """频率响应测试"""
        print("\n" + "=" * 60)
        print("📊 频率响应测试")
        print("=" * 60)
        
        # 获取PFB滤波器响应
        freqs, pfb_response_db = self.pfb_analysis.get_filter_response(num_points=2048)
        
        # 计算STFT窗响应
        window = signal.get_window('hann', self.fft_size)
        stft_response = np.fft.fft(window, 2048)
        stft_response_db = 20 * np.log10(np.abs(stft_response) + 1e-12)
        stft_response_db = stft_response_db[:len(freqs)]
        
        # 计算旁瓣衰减
        pfb_sidelobe = np.max(pfb_response_db[int(len(freqs)*0.2):])
        stft_sidelobe = np.max(stft_response_db[int(len(freqs)*0.2):])
        
        print(f"\n📈 旁瓣衰减:")
        print(f"   PFB: {pfb_sidelobe:.2f} dB")
        print(f"   STFT (Hann): {stft_sidelobe:.2f} dB")
        print(f"   差异: {pfb_sidelobe - stft_sidelobe:.2f} dB")
        
        results = {
            'freqs': freqs,
            'pfb_response': pfb_response_db,
            'stft_response': stft_response_db,
            'pfb_sidelobe': pfb_sidelobe,
            'stft_sidelobe': stft_sidelobe
        }
        
        # 绘图
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(freqs, pfb_response_db, label='PFB (Kaiser)', linewidth=2)
        plt.plot(freqs, stft_response_db, label='STFT (Hann)', linewidth=2, alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Frequency Response')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(freqs, pfb_response_db, label='PFB', linewidth=2)
        plt.plot(freqs, stft_response_db, label='STFT', linewidth=2, alpha=0.7)
        plt.ylim([-80, 5])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Frequency Response (Zoomed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('frequency_response_comparison.png', dpi=150)
        print(f"\n✅ 频率响应对比图已保存: frequency_response_comparison.png")
        
        return results
    
    def performance_test(self, signal: np.ndarray, num_iterations: int = 100) -> Dict:
        """性能测试"""
        print("\n" + "=" * 60)
        print("📊 性能测试")
        print("=" * 60)
        
        results = {}
        
        # NumPy PFB
        print(f"\n🔄 NumPy PFB ({num_iterations} 次迭代)...")
        start = time.time()
        for _ in range(num_iterations):
            mag, ph = self.pfb_analysis.process(signal)
            reconstructed = self.pfb_synthesis.process(mag, ph)
        numpy_pfb_time = (time.time() - start) / num_iterations * 1000  # ms
        
        results['numpy_pfb'] = {
            'avg_time_ms': numpy_pfb_time,
            'realtime_factor': numpy_pfb_time / 1000 * self.sample_rate
        }
        
        print(f"   平均时间: {numpy_pfb_time:.2f} ms")
        print(f"   实时因子: {results['numpy_pfb']['realtime_factor']:.2f}x")
        
        # NumPy STFT
        print(f"\n🔄 NumPy STFT ({num_iterations} 次迭代)...")
        start = time.time()
        for _ in range(num_iterations):
            mag, ph = compute_stft(signal, self.fft_size, self.hop_size)
            reconstructed = compute_istft(mag, ph, self.hop_size)
        numpy_stft_time = (time.time() - start) / num_iterations * 1000  # ms
        
        results['numpy_stft'] = {
            'avg_time_ms': numpy_stft_time,
            'realtime_factor': numpy_stft_time / 1000 * self.sample_rate
        }
        
        print(f"   平均时间: {numpy_stft_time:.2f} ms")
        print(f"   实时因子: {results['numpy_stft']['realtime_factor']:.2f}x")
        
        # PyTorch PFB (GPU)
        if self.device == 'cuda':
            print(f"\n🔄 PyTorch PFB GPU ({num_iterations} 次迭代)...")
            audio_torch = torch.from_numpy(signal).unsqueeze(0).float().cuda()
            
            with torch.no_grad():
                start = time.time()
                for _ in range(num_iterations):
                    reconstructed_torch = self.pfb_pytorch(audio_torch)
                pytorch_pfb_time = (time.time() - start) / num_iterations * 1000  # ms
            
            results['pytorch_pfb_gpu'] = {
                'avg_time_ms': pytorch_pfb_time,
                'realtime_factor': pytorch_pfb_time / 1000 * self.sample_rate
            }
            
            print(f"   平均时间: {pytorch_pfb_time:.2f} ms")
            print(f"   实时因子: {results['pytorch_pfb_gpu']['realtime_factor']:.2f}x")
        
        # 对比
        print(f"\n📈 对比:")
        print(f"   NumPy PFB vs NumPy STFT: {numpy_pfb_time / numpy_stft_time:.2f}x")
        if self.device == 'cuda':
            print(f"   PyTorch GPU vs NumPy STFT: {numpy_stft_time / pytorch_pfb_time:.2f}x 加速")
        
        return results
    
    def spectrogram_comparison(self, signal: np.ndarray):
        """语谱图对比"""
        print("\n" + "=" * 60)
        print("📊 语谱图对比")
        print("=" * 60)
        
        # PFB语谱图
        pfb_mag, _ = self.pfb_analysis.process(signal)
        
        # STFT语谱图
        stft_mag, _ = compute_stft(signal, self.fft_size, self.hop_size)
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # PFB幅度
        im1 = axes[0, 0].imshow(pfb_mag.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('PFB Magnitude Spectrogram')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Frequency Bin')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # STFT幅度
        im2 = axes[0, 1].imshow(stft_mag, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('STFT Magnitude Spectrogram')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Frequency Bin')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # PFB对数幅度
        im3 = axes[1, 0].imshow(20 * np.log10(pfb_mag.T + 1e-12), aspect='auto', 
                                   origin='lower', cmap='viridis', vmin=-80)
        axes[1, 0].set_title('PFB Log Magnitude (dB)')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Frequency Bin')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # STFT对数幅度
        im4 = axes[1, 1].imshow(20 * np.log10(stft_mag + 1e-12), aspect='auto', 
                                   origin='lower', cmap='viridis', vmin=-80)
        axes[1, 1].set_title('STFT Log Magnitude (dB)')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Frequency Bin')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('spectrogram_comparison.png', dpi=150)
        print(f"\n✅ 语谱图对比已保存: spectrogram_comparison.png")


def generate_test_signals(sample_rate: float, duration: float) -> Dict[str, np.ndarray]:
    """生成测试信号"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    signals = {}
    
    # 多频正弦波
    signals['multitone'] = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 1000 * t) +
        0.2 * np.sin(2 * np.pi * 3000 * t)
    )
    
    # 线性调频信号
    signals['chirp'] = signal.chirp(t, f0=100, f1=4000, t1=duration, method='linear')
    
    # 语音模拟信号（共振峰）
    f1, f2, f3 = 500, 1500, 2500  # 共振峰频率
    signals['speech_like'] = (
        0.5 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.2 * np.sin(2 * np.pi * f3 * t) +
        0.05 * np.random.randn(len(t))
    )
    
    # 宽带噪声
    signals['noise'] = 0.5 * np.random.randn(len(t))
    
    return signals


if __name__ == "__main__":
    print("=" * 60)
    print("PFB vs STFT 全面对比测试")
    print("=" * 60)
    
    # 配置
    sample_rate = 16000
    duration = 2.0
    fft_size = 128
    hop_size = 64
    filter_length = 768
    
    # 创建对比器
    comparer = PFBvsSTFTComparer(
        fft_size=fft_size,
        hop_size=hop_size,
        filter_length=filter_length,
        sample_rate=sample_rate
    )
    
    # 生成测试信号
    print(f"\n📊 生成测试信号...")
    signals = generate_test_signals(sample_rate, duration)
    
    # 对每个信号进行测试
    for name, sig in signals.items():
        print(f"\n" + "=" * 60)
        print(f"📊 测试信号: {name}")
        print("=" * 60)
        
        # 完美重建测试
        reconstruction_results = comparer.perfect_reconstruction_test(sig)
        
        # 语谱图对比（只对第一个信号）
        if name == list(signals.keys())[0]:
            comparer.spectrogram_comparison(sig)
    
    # 频率响应测试
    freq_response_results = comparer.frequency_response_test()
    
    # 性能测试（使用多频信号）
    performance_results = comparer.performance_test(signals['multitone'], num_iterations=100)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    print(f"\n📈 完美重建质量:")
    print(f"   PFB平均SNR: {np.mean([r['pfb']['snr'] for r in [reconstruction_results]]):.2f} dB")
    print(f"   STFT平均SNR: {np.mean([r['stft']['snr'] for r in [reconstruction_results]]):.2f} dB")
    
    print(f"\n📈 旁瓣衰减:")
    print(f"   PFB: {freq_response_results['pfb_sidelobe']:.2f} dB")
    print(f"   STFT: {freq_response_results['stft_sidelobe']:.2f} dB")
    print(f"   PFB优势: {freq_response_results['stft_sidelobe'] - freq_response_results['pfb_sidelobe']:.2f} dB")
    
    print(f"\n📈 性能:")
    print(f"   NumPy PFB: {performance_results['numpy_pfb']['avg_time_ms']:.2f} ms")
    print(f"   NumPy STFT: {performance_results['numpy_stft']['avg_time_ms']:.2f} ms")
    if 'pytorch_pfb_gpu' in performance_results:
        print(f"   PyTorch GPU PFB: {performance_results['pytorch_pfb_gpu']['avg_time_ms']:.2f} ms")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
