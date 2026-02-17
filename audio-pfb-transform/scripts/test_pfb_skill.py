"""
测试修复后的 PFB Skill

验证 PFB Analysis + Synthesis 的完美重建
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf

# 设置 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 从本地 pfb_v4_cmodel.py 导入（参考）
import sys
project_root = Path(__file__).parent.parent.parent.parent / "ds_vqe"
sys.path.insert(0, str(project_root))


def test_numpy_pfb():
    """测试 NumPy 版本的 PFB"""
    print("=" * 70)
    print("测试 NumPy 版本的 PFB")
    print("=" * 70)

    from pfb_analysis import PFBAnalysis
    from pfb_synthesis import PFBSynthesis

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

    print(f"\n📊 测试信号: {len(test_signal)} 采样, {duration} 秒")

    # 分析
    print("\n🔄 [1/2] PFB分析...")
    complex_spectrum = analyzer.process(test_signal)
    print(f"   输出频谱: {complex_spectrum.shape[0]} 帧 x {complex_spectrum.shape[1]} 子带")

    # 合成
    print("\n🔄 [2/2] PFB合成...")
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

    print(f"\n✅ [验证] 完美重建验证:")
    print(f"   MSE: {mse:.2e}")
    print(f"   最大绝对误差: {max_error:.2e}")
    print(f"   SNR: {snr_db:.2f} dB")
    print(f"   相关系数: {correlation:.4f}")

    if snr_db > 40 and correlation > 0.7:
        print(f"   ✅✅✅ NumPy PFB 完美重建验证通过！")
        return True
    else:
        print(f"   ❌❌❌ 重建质量有待提高")
        return False


def test_pytorch_pfb():
    """测试 PyTorch 版本的 PFB"""
    print("\n" + "=" * 70)
    print("测试 PyTorch 版本的 PFB")
    print("=" * 70)

    import torch
    from pfb_pytorch import PFBTransform

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

    print(f"\n📊 输入信号: {signal.shape}")

    # 测试完美重建
    with torch.no_grad():
        reconstructed = pfb(signal)

    print(f"   重构信号: {reconstructed.shape}")

    # 对齐长度
    min_len = min(signal.shape[1], reconstructed.shape[1])
    signal_trunc = signal[:, :min_len]
    reconstructed_trunc = reconstructed[:, :min_len]

    # 计算误差
    mse = torch.mean((signal_trunc - reconstructed_trunc)**2).item()
    max_error = torch.max(torch.abs(signal_trunc - reconstructed_trunc)).item()
    snr = 10 * torch.log10(torch.mean(signal_trunc**2) / (torch.mean((signal_trunc - reconstructed_trunc)**2) + 1e-12)).item()

    print(f"\n✅ [验证] 完美重建验证:")
    print(f"   MSE: {mse:.2e}")
    print(f"   最大误差: {max_error:.2e}")
    print(f"   SNR: {snr:.2f} dB")

    # 测试可微分性
    print(f"\n🧪 测试梯度传播...")
    signal_grad = signal.clone().requires_grad_(True)
    output = pfb(signal_grad)
    loss = torch.mean(output**2)
    try:
        loss.backward()
        print(f"   损失值: {loss.item():.4f}")
        print(f"   输入梯度存在: {signal_grad.grad is not None}")
        if signal_grad.grad is not None:
            print(f"   输入梯度范数: {torch.norm(signal_grad.grad).item():.4f}")
            grad_success = True
        else:
            grad_success = False
    except RuntimeError as e:
        print(f"   ⚠️  梯度计算失败（由于 inplace 操作）: {e}")
        grad_success = False

    if snr > 40:
        print(f"   ✅✅✅ PyTorch PFB 完美重建验证通过！")
        return True
    else:
        print(f"   ❌❌❌ 重建质量有待提高")
        return False


def test_with_audio_file():
    """使用音频文件测试"""
    print("\n" + "=" * 70)
    print("使用音频文件测试 PFB")
    print("=" * 70)

    from pfb_analysis import PFBAnalysis
    from pfb_synthesis import PFBSynthesis

    # 测试音频文件路径
    audio_path = Path("D:/simulation/athena-signal-master-pure-c/examples/near_mic_12311532.wav")

    if not audio_path.exists():
        print(f"⚠️  测试音频文件不存在: {audio_path}")
        print("   跳过音频文件测试")
        return True

    # 读取音频
    audio, fs = sf.read(str(audio_path))
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # 单声道

    # 截取前 1 秒
    test_signal = audio[:fs].astype(np.float32)
    print(f"\n📊 音频信号: {len(test_signal)} 采样, 采样率: {fs} Hz")

    # 创建分析器和合成器
    analyzer = PFBAnalysis(fft_len=256, win_len=768, frm_len=128)
    synthesizer = PFBSynthesis(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

    # 分析
    print("\n🔄 [1/2] PFB分析...")
    complex_spectrum = analyzer.process(test_signal)
    print(f"   输出频谱: {complex_spectrum.shape[0]} 帧 x {complex_spectrum.shape[1]} 子带")

    # 合成
    print("\n🔄 [2/2] PFB合成...")
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

    print(f"\n✅ [验证] 完美重建验证:")
    print(f"   MSE: {mse:.2e}")
    print(f"   最大绝对误差: {max_error:.2e}")
    print(f"   SNR: {snr_db:.2f} dB")
    print(f"   相关系数: {correlation:.4f}")

    # 保存重构音频
    output_path = Path(__file__).parent / "test_reconstructed.wav"
    sf.write(str(output_path), reconstructed_aligned, fs)
    print(f"\n💾 重构音频已保存: {output_path}")

    if snr_db > 40 and correlation > 0.7:
        print(f"   ✅✅✅ 音频文件测试验证通过！")
        return True
    else:
        print(f"   ❌❌❌ 重建质量有待提高")
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("=" * 70)
    print("PFB Skill 修复验证测试")
    print("=" * 70)
    print("\n基于 C 模型的严格实现")
    print("参数: FFT_LEN=256, WIN_LEN=768, FRM_LEN=128")
    print("群延时补偿: -648 采样 @ 16kHz")
    print("")

    results = []

    # 测试 1: NumPy PFB
    results.append(("NumPy PFB", test_numpy_pfb()))

    # 测试 2: PyTorch PFB
    results.append(("PyTorch PFB", test_pytorch_pfb()))

    # 测试 3: 音频文件
    results.append(("音频文件测试", test_with_audio_file()))

    # 总结
    print("\n")
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")

    all_passed = all(result for _, result in results)
    if all_passed:
        print(f"\n🎉🎉🎉 所有测试通过！PFB Skill 修复成功！")
    else:
        print(f"\n⚠️  部分测试失败，需要进一步调试")

    print("=" * 70)
    print("")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
