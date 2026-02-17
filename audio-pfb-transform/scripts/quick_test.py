"""
快速测试 NumPy PFB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 设置 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np

from pfb_analysis import PFBAnalysis
from pfb_synthesis import PFBSynthesis

print("=" * 60)
print("NumPy PFB 快速测试")
print("=" * 60)

analyzer = PFBAnalysis(fft_len=256, win_len=768, frm_len=128)
synthesizer = PFBSynthesis(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

# 生成测试信号
fs = 16000
duration = 0.5
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
test_signal = (
    0.5 * np.sin(2 * np.pi * 440 * t) +
    0.3 * np.sin(2 * np.pi * 1000 * t) +
    0.2 * np.sin(2 * np.pi * 3000 * t)
).astype(np.float32)

print(f"\n测试信号: {len(test_signal)} 采样")

# 分析
print("\n分析...")
complex_spectrum = analyzer.process(test_signal)
print(f"   输出频谱: {complex_spectrum.shape}")

# 合成
print("\n合成...")
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
correlation = np.corrcoef(test_signal_aligned, reconstructed_aligned)[0, 1]

print(f"\n完美重建验证:")
print(f"   MSE: {mse:.2e}")
print(f"   最大误差: {max_error:.2e}")
print(f"   SNR: {snr_db:.2f} dB")
print(f"   相关系数: {correlation:.4f}")

if snr_db > 40 and correlation > 0.7:
    print(f"   ✅✅✅ NumPy PFB 完美重建验证通过！")
else:
    print(f"   ❌❌❌ 重建质量有待提高")

print("=" * 60)
