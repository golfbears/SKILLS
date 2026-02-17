"""
PFB Skill 使用示例

展示如何使用修复后的 PFB Skill 进行音频分析合成
"""

import sys
from pathlib import Path

# 设置 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
from pfb_analysis import PFBAnalysis
from pfb_synthesis import PFBSynthesis


def example_numpy_pfb():
    """NumPy PFB 使用示例"""
    print("=" * 70)
    print("NumPy PFB 使用示例")
    print("=" * 70)

    # 创建分析器和合成器
    analyzer = PFBAnalysis(fft_len=256, win_len=768, frm_len=128)
    synthesizer = PFBSynthesis(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

    # 生成测试音频
    fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 混合多个频率的正弦波
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +      # A4
        0.3 * np.sin(2 * np.pi * 1000 * t) +    # 1kHz
        0.2 * np.sin(2 * np.pi * 3000 * t)      # 3kHz
    ).astype(np.float32)

    print(f"\n输入音频:")
    print(f"   采样率: {fs} Hz")
    print(f"   时长: {duration} 秒")
    print(f"   采样点数: {len(audio)}")

    # 分析
    print(f"\n[1/3] PFB 分析...")
    complex_spectrum = analyzer.process(audio)
    print(f"   频谱形状: {complex_spectrum.shape}")
    print(f"   帧数: {complex_spectrum.shape[0]}")
    print(f"   子带数: {complex_spectrum.shape[1]}")

    # 可以在频域进行处理（例如频谱掩码、滤波等）
    # 这里我们直接重建（不进行任何处理）

    # 合成
    print(f"\n[2/3] PFB 合成...")
    reconstructed = synthesizer.process(complex_spectrum)
    print(f"   重构音频长度: {len(reconstructed)} 采样")

    # 对齐长度
    min_len = min(len(audio), len(reconstructed))
    audio_aligned = audio[:min_len]
    reconstructed_aligned = reconstructed[:min_len]

    # 评估重建质量
    print(f"\n[3/3] 评估重建质量...")
    error = audio_aligned - reconstructed_aligned
    mse = np.mean(error ** 2)
    signal_power = np.mean(audio_aligned ** 2)
    snr_db = 10 * np.log10(signal_power / (mse + 1e-12))
    correlation = np.corrcoef(audio_aligned, reconstructed_aligned)[0, 1]

    print(f"   MSE: {mse:.2e}")
    print(f"   SNR: {snr_db:.2f} dB")
    print(f"   相关系数: {correlation:.4f}")

    if snr_db > 40 and correlation > 0.7:
        print(f"\n✅ 完美重建验证通过！")
    else:
        print(f"\n⚠️  重建质量有待提高")

    # 保存音频
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    sf.write(str(output_dir / "input_audio.wav"), audio, fs)
    sf.write(str(output_dir / "reconstructed_audio.wav"), reconstructed_aligned, fs)

    print(f"\n💾 音频已保存到: {output_dir}")
    print(f"   - input_audio.wav: 原始音频")
    print(f"   - reconstructed_audio.wav: 重构音频")

    print("\n" + "=" * 70)


def example_with_real_audio():
    """使用真实音频文件的示例"""
    print("\n" + "=" * 70)
    print("使用真实音频文件的示例")
    print("=" * 70)

    # 音频文件路径
    audio_path = Path("D:/simulation/athena-signal-master-pure-c/examples/near_mic_12311532.wav")

    if not audio_path.exists():
        print(f"\n⚠️  音频文件不存在: {audio_path}")
        print("   跳过此示例")
        return

    # 读取音频
    audio, fs = sf.read(str(audio_path))
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # 单声道

    # 截取前 2 秒
    audio = audio[:fs*2].astype(np.float32)

    print(f"\n输入音频:")
    print(f"   文件: {audio_path.name}")
    print(f"   采样率: {fs} Hz")
    print(f"   时长: {len(audio)/fs:.2f} 秒")

    # 创建分析器和合成器
    analyzer = PFBAnalysis(fft_len=256, win_len=768, frm_len=128)
    synthesizer = PFBSynthesis(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

    # 分析和合成
    print(f"\n[1/2] PFB 分析...")
    complex_spectrum = analyzer.process(audio)
    print(f"   频谱形状: {complex_spectrum.shape}")

    print(f"\n[2/2] PFB 合成...")
    reconstructed = synthesizer.process(complex_spectrum)
    print(f"   重构音频长度: {len(reconstructed)} 采样")

    # 对齐长度
    min_len = min(len(audio), len(reconstructed))
    audio_aligned = audio[:min_len]
    reconstructed_aligned = reconstructed[:min_len]

    # 评估重建质量
    error = audio_aligned - reconstructed_aligned
    mse = np.mean(error ** 2)
    signal_power = np.mean(audio_aligned ** 2)
    snr_db = 10 * np.log10(signal_power / (mse + 1e-12))
    correlation = np.corrcoef(audio_aligned, reconstructed_aligned)[0, 1]

    print(f"\n重建质量:")
    print(f"   MSE: {mse:.2e}")
    print(f"   SNR: {snr_db:.2f} dB")
    print(f"   相关系数: {correlation:.4f}")

    # 保存音频
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    sf.write(str(output_dir / "real_audio_reconstructed.wav"), reconstructed_aligned, fs)

    print(f"\n💾 重构音频已保存: {output_dir / 'real_audio_reconstructed.wav'}")

    print("\n" + "=" * 70)


def example_pytorch_pfb():
    """PyTorch PFB 使用示例"""
    print("\n" + "=" * 70)
    print("PyTorch PFB 使用示例")
    print("=" * 70)

    try:
        import torch
        from pfb_pytorch import PFBTransform
    except ImportError:
        print("\n⚠️  PyTorch 未安装，跳过此示例")
        return

    # 创建 PFB 变换
    pfb = PFBTransform(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pfb = pfb.to(device)

    print(f"\n使用设备: {device}")

    # 生成测试音频
    fs = 16000
    num_samples = fs * 1  # 1 秒
    t = torch.linspace(0, 1, num_samples, device=device)

    audio = (
        0.5 * torch.sin(2 * np.pi * 440 * t) +
        0.3 * torch.sin(2 * np.pi * 1000 * t) +
        0.2 * torch.sin(2 * np.pi * 3000 * t)
    ).unsqueeze(0)  # 添加 batch 维度

    print(f"\n输入音频:")
    print(f"   形状: {audio.shape}")
    print(f"   采样率: {fs} Hz")
    print(f"   采样点数: {num_samples}")

    # 完美的分析-合成循环
    with torch.no_grad():
        reconstructed = pfb(audio)

    print(f"\n重构音频:")
    print(f"   形状: {reconstructed.shape}")

    # 对齐长度
    min_len = min(audio.shape[1], reconstructed.shape[1])
    audio_trunc = audio[:, :min_len]
    reconstructed_trunc = reconstructed[:, :min_len]

    # 评估重建质量
    mse = torch.mean((audio_trunc - reconstructed_trunc) ** 2).item()
    snr = 10 * torch.log10(torch.mean(audio_trunc ** 2) / (mse + 1e-12)).item()

    print(f"\n重建质量:")
    print(f"   MSE: {mse:.2e}")
    print(f"   SNR: {snr:.2f} dB")

    if snr > 40:
        print(f"\n✅ PyTorch PFB 完美重建验证通过！")
    else:
        print(f"\n⚠️  重建质量有待提高")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 运行所有示例
    example_numpy_pfb()
    example_with_real_audio()
    example_pytorch_pfb()

    print("\n" + "=" * 70)
    print("所有示例运行完成！")
    print("=" * 70)
    print("\n提示:")
    print("1. NumPy 版本适用于离线处理")
    print("2. PyTorch 版本适用于深度学习训练")
    print("3. 两种实现都支持群延时补偿（-648 采样）")
    print("=" * 70)
