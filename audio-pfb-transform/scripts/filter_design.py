"""
PFB滤波器设计工具

提供Kaiser窗原型滤波器的设计、保存和加载功能。
"""

import numpy as np
from scipy.signal import kaiserord, firwin, freqz
from scipy.fft import fftfreq
import json
from pathlib import Path
from typing import Tuple, Optional
import sys
import io

# 设置UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class KiserFilterDesigner:
    """Kaiser窗滤波器设计器"""
    
    def __init__(
        self,
        sample_rate: float = 16000,
        cutoff_ratio: float = 0.5,
        ripple_db: float = 60.0,
        transition_width_ratio: float = 0.1
    ):
        """
        初始化滤波器设计器
        
        Args:
            sample_rate: 采样率(Hz)
            cutoff_ratio: 截止频率比率(相对于Nyquist)
            ripple_db: 允许的波纹(dB)
            transition_width_ratio: 过渡带宽度比率
        """
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        self.cutoff = self.nyquist * cutoff_ratio
        self.ripple_db = ripple_db
        self.transition_width = self.nyquist * transition_width_ratio
    
    def design_filter(self, filter_length: int) -> np.ndarray:
        """
        设计Kaiser窗FIR滤波器
        
        Args:
            filter_length: 滤波器长度(必须为奇数)
            
        Returns:
            滤波器系数
        """
        if filter_length % 2 == 0:
            filter_length += 1
        
        # 计算Kaiser窗参数
        taps, beta = kaiserord(
            self.ripple_db,
            self.transition_width / self.nyquist
        )
        
        # 限制滤波器长度
        taps = min(taps, filter_length)
        if taps % 2 == 0:
            taps -= 1
        
        # 设计FIR滤波器
        h = firwin(
            taps,
            self.cutoff,
            window=('kaiser', beta),
            pass_zero='lowpass',
            fs=self.sample_rate
        )
        
        return h
    
    def analyze_filter(self, h: np.ndarray, plot: bool = False) -> dict:
        """
        分析滤波器特性
        
        Args:
            h: 滤波器系数
            plot: 是否绘制频率响应
            
        Returns:
            滤波器特性字典
        """
        # 频率响应
        w, H = freqz(h, fs=self.sample_rate)
        
        # 计算关键指标
        H_mag = np.abs(H)
        H_mag_db = 20 * np.log10(H_mag + 1e-12)
        
        # 通带波纹
        passband_idx = np.where(w <= self.cutoff)[0]
        passband_ripple = np.max(H_mag_db[passband_idx]) - np.min(H_mag_db[passband_idx])
        
        # 阻带衰减
        stopband_freq = self.cutoff * 1.1
        stopband_idx = np.where(w >= stopband_freq)[0]
        stopband_attenuation = np.max(H_mag_db[stopband_idx])
        
        results = {
            'filter_length': len(h),
            'passband_ripple_db': passband_ripple,
            'stopband_attenuation_db': stopband_attenuation,
            'cutoff_frequency_hz': self.cutoff,
            'nyquist_frequency_hz': self.nyquist
        }
        
        if plot:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 幅频响应
            ax1.plot(w, H_mag_db)
            ax1.axvline(self.cutoff, color='r', linestyle='--', label=f'Cutoff ({self.cutoff:.1f} Hz)')
            ax1.axhline(-self.ripple_db, color='g', linestyle=':', label=f'-{self.ripple_db} dB')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Magnitude (dB)')
            ax1.set_title('Frequency Response')
            ax1.grid(True)
            ax1.legend()
            
            # 相频响应
            ax2.plot(w, np.unwrap(np.angle(H)))
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Phase (radians)')
            ax2.set_title('Phase Response')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return results


def compute_kaiser_filter(
    fft_size: int,
    filter_length: int,
    sample_rate: float = 16000,
    kaiser_beta: float = 12.0,
    cutoff_ratio: float = 0.5
) -> np.ndarray:
    """
    计算用于PFB的Kaiser窗原型滤波器
    
    Args:
        fft_size: FFT大小
        filter_length: 滤波器长度(通常=fft_size*6)
        sample_rate: 采样率
        kaiser_beta: Kaiser窗beta参数
        cutoff_ratio: 截止频率比率
        
    Returns:
        原型滤波器系数 (filter_length,)
    """
    if filter_length % 2 == 0:
        filter_length += 1
    
    # 截止频率(归一化到Nyquist)
    cutoff = cutoff_ratio
    
    # 生成时间序列
    t = np.arange(filter_length) - (filter_length - 1) / 2
    t = t / filter_length
    
    # 理想低通滤波器脉冲响应
    h_ideal = 2 * cutoff * np.sinc(2 * cutoff * t)
    
    # Kaiser窗
    from scipy.special import i0
    alpha = kaiser_beta * np.pi
    window = i0(alpha * np.sqrt(1 - (2 * t)**2)) / i0(alpha)
    
    # 应用窗函数
    h = h_ideal * window
    
    # 归一化以保持能量
    h = h / np.sum(np.abs(h))
    
    return h


def compute_polyphase_filters(
    prototype_filter: np.ndarray,
    num_subbands: int
) -> np.ndarray:
    """
    将原型滤波器分解为多相分量
    
    Args:
        prototype_filter: 原型滤波器系数 (L,)
        num_subbands: 子带数量(=fft_size)
        
    Returns:
        多相滤波器组 (num_subbands, L//num_subbands)
    """
    L = len(prototype_filter)
    
    # 确保能被num_subbands整除，不补零
    adjusted_length = (L // num_subbands) * num_subbands
    h_truncated = prototype_filter[:adjusted_length]
    
    # 重塑为多相形式
    polyphase = h_truncated.reshape(num_subbands, -1)
    
    return polyphase


def save_filter_coefficients(
    h: np.ndarray,
    config: dict,
    filepath: str
):
    """
    保存滤波器系数和配置
    
    Args:
        h: 滤波器系数
        config: 配置字典
        filepath: 保存路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存系数
    np.save(filepath, h)
    
    # 保存配置
    config_path = filepath.with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 滤波器系数已保存到: {filepath}")
    print(f"✅ 配置已保存到: {config_path}")


def load_filter_coefficients(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    加载滤波器系数和配置
    
    Args:
        filepath: 滤波器系数文件路径
        
    Returns:
        (滤波器系数, 配置字典)
    """
    filepath = Path(filepath)
    
    # 加载系数
    h = np.load(filepath)
    
    # 加载配置
    config_path = filepath.with_suffix('.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return h, config


# 预设配置
PRESETS = {
    '16k_128_64_768': {
        'sample_rate': 16000,
        'fft_size': 128,
        'hop_size': 64,
        'filter_length': 768,
        'kaiser_beta': 12.0,
        'cutoff_ratio': 0.5
    },
    '48k_256_128_1536': {
        'sample_rate': 48000,
        'fft_size': 256,
        'hop_size': 128,
        'filter_length': 1536,
        'kaiser_beta': 12.0,
        'cutoff_ratio': 0.5
    },
    '16k_64_32_384': {
        'sample_rate': 16000,
        'fft_size': 64,
        'hop_size': 32,
        'filter_length': 384,
        'kaiser_beta': 12.0,
        'cutoff_ratio': 0.5
    }
}


def generate_preset_filters(output_dir: str = "./assets/filter_coefficients"):
    """
    生成所有预设滤波器
    
    Args:
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, config in PRESETS.items():
        print(f"\n📊 生成预设滤波器: {name}")
        print(f"   配置: {config}")
        
        h = compute_kaiser_filter(
            fft_size=config['fft_size'],
            filter_length=config['filter_length'],
            sample_rate=config['sample_rate'],
            kaiser_beta=config['kaiser_beta'],
            cutoff_ratio=config['cutoff_ratio']
        )
        
        filepath = output_dir / f"kaiser_{name}.npy"
        save_filter_coefficients(h, config, str(filepath))
        
        # 分析滤波器特性
        designer = KiserFilterDesigner(
            sample_rate=config['sample_rate'],
            cutoff_ratio=config['cutoff_ratio'],
            ripple_db=60.0
        )
        results = designer.analyze_filter(h, plot=False)
        print(f"   通带波纹: {results['passband_ripple_db']:.2f} dB")
        print(f"   阻带衰减: {results['stopband_attenuation_db']:.2f} dB")


if __name__ == "__main__":
    # 生成预设滤波器
    print("=" * 60)
    print("PFB滤波器系数生成器")
    print("=" * 60)
    
    generate_preset_filters()
    
    print("\n" + "=" * 60)
    print("✅ 所有预设滤波器生成完成！")
    print("=" * 60)
