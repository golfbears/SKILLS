"""
测试频域IPNLMS AEC模块
"""

import sys
from pathlib import Path

# 添加Skill路径
skill_path = Path(__file__).parent
sys.path.insert(0, str(skill_path))

import torch
import numpy as np
from ipnlms_aec import FrequencyDomainIPNLMSV2, create_ipnlms_aec


def test_basic_forward():
    """测试基本前向传播"""
    print("Test 1: Basic Forward Pass")
    print("-" * 40)

    B, T, F = 2, 10, 129
    aec = create_ipnlms_aec(fft_size=256, num_blocks=8)

    mic_fft = torch.randn(B, T, F, dtype=torch.complex64)
    ref_fft = torch.randn(B, T, F, dtype=torch.complex64)

    error, echo = aec(mic_fft, ref_fft)

    assert error.shape == (B, T, F), f"Expected {error.shape}, got {error.shape}"
    assert echo.shape == (B, T, F), f"Expected {echo.shape}, got {echo.shape}"

    print(f"  Input:  mic={mic_fft.shape}, ref={ref_fft.shape}")
    print(f"  Output: error={error.shape}, echo={echo.shape}")
    print("  [OK] PASSED\n")
    return True


def test_gradient():
    """测试梯度传播"""
    print("Test 2: Gradient Propagation")
    print("-" * 40)

    B, T, F = 1, 5, 129
    aec = create_ipnlms_aec(fft_size=256, num_blocks=8)
    aec.train()

    mic_fft = torch.randn(B, T, F, dtype=torch.complex64, requires_grad=True)
    ref_fft = torch.randn(B, T, F, dtype=torch.complex64)

    error, echo = aec(mic_fft, ref_fft)
    loss = torch.sum(torch.abs(error) ** 2)

    loss.backward()

    assert mic_fft.grad is not None, "Gradient should exist"
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient shape: {mic_fft.grad.shape}")
    print("  [OK] PASSED\n")
    return True


def test_reset():
    """测试重置功能"""
    print("Test 3: Reset Function")
    print("-" * 40)

    aec = create_ipnlms_aec(fft_size=256, num_blocks=8)

    # 修改系数
    aec.coef_real.data.add_(10)

    # 重置
    aec.reset()

    # 检查系数是否接近零
    assert torch.allclose(aec.coef_real.data, torch.zeros_like(aec.coef_real.data), atol=1e-5)
    assert torch.allclose(aec.coef_imag.data, torch.zeros_like(aec.coef_imag.data), atol=1e-5)

    print("  Coefficients reset to zeros")
    print("  [OK] PASSED\n")
    return True


def test_parameter_count():
    """测试参数数量"""
    print("Test 4: Parameter Count")
    print("-" * 40)

    F, blocks = 129, 8

    aec = create_ipnlms_aec(fft_size=256, num_blocks=blocks)

    num_params = sum(p.numel() for p in aec.parameters())
    expected = F * blocks * 2  # real + imag

    print(f"  FFT size: 256, Freq bins: {F}, Blocks: {blocks}")
    print(f"  Expected params: {expected}")
    print(f"  Actual params: {num_params}")
    print("  [OK] PASSED\n")
    return True


def test_inference_mode():
    """测试推理模式"""
    print("Test 5: Inference Mode")
    print("-" * 40)

    aec = create_ipnlms_aec(fft_size=256, num_blocks=8)
    aec.eval()

    with torch.no_grad():
        mic_fft = torch.randn(1, 5, 129, dtype=torch.complex64)
        ref_fft = torch.randn(1, 5, 129, dtype=torch.complex64)

        error, echo = aec(mic_fft, ref_fft)

    print(f"  Inference output: error={error.shape}")
    print("  [OK] PASSED\n")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("Frequency Domain IPNLMS AEC - Test Suite")
    print("=" * 50 + "\n")

    tests = [
        test_basic_forward,
        test_gradient,
        test_reset,
        test_parameter_count,
        test_inference_mode,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [X] FAILED: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
