# PFB Skill 修复说明

## 概述
PFB Skill 已使用 `pfb_v4_cmodel.py` 中的验证正确的实现进行修复。

## 核心参数
- FFT_LEN = 256
- WIN_LEN = 768
- FRM_LEN = 128
- Ppf_tap = 3
- Ppf_decm = 6
- Scale = -256.0
- 群延时补偿 = -648 采样 @ 16kHz

## 使用方法

### NumPy 版本
```python
from pfb_analysis import PFBAnalysis
from pfb_synthesis import PFBSynthesis

# 创建分析器和合成器
analyzer = PFBAnalysis(fft_len=256, win_len=768, frm_len=128)
synthesizer = PFBSynthesis(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

# 分析
complex_spectrum = analyzer.process(audio_signal)

# 合成（自动应用群延时补偿）
reconstructed = synthesizer.process(complex_spectrum)
```

### PyTorch 版本
```python
from pfb_pytorch import PFBTransform

# 创建完整的 PFB 变换
pfb = PFBTransform(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

# 前向传播（用于测试完美重建）
reconstructed = pfb(audio_tensor)

# 分析和合成也可以分开使用
from pfb_pytorch import PFBAnalysisLayer, PFBSynthesisLayer

analyzer = PFBAnalysisLayer(fft_len=256, win_len=768, frm_len=128)
synthesizer = PFBSynthesisLayer(fft_len=256, win_len=768, frm_len=128, scale=-256.0)

spectrum = analyzer(audio_tensor)
reconstructed = synthesizer(spectrum)
```

## 注意事项

1. **群延时补偿**: PFB 实现有 -648 采样的群延时，合成时会自动补偿（16kHz 采样率下为 40.5ms）

2. **Scale 因子**: 使用 -256.0 的缩放因子以确保正确的功率匹配

3. **滤波器系数**: 需要使用正确的滤波器系数文件 `assets/pfb_filter_coef_768.npy`

4. **PyTorch 可微分性**: PyTorch 版本支持梯度计算，可以用于深度学习训练

## 测试

运行快速测试：
```bash
python scripts/quick_test.py
```

运行完整测试：
```bash
python scripts/test_pfb_skill.py
```

## 实现细节

### 分析器
- 严格遵循 C 模型逻辑 (dios_ssp_share_subband.c)
- 使用 RFFT 转换
- 多相滤波器实现

### 合成器
- 使用 IRFFT 转换
- 多相合成，累积缓冲区
- 自动应用 -648 采样群延时补偿
- 使用 -256.0 缩放因子

### PyTorch 版本
- 基于 NumPy 实现的严格移植
- 支持 GPU 加速
- 可用于深度学习训练
- 梯度计算功能

## 故障排除

如果重建质量不佳，请检查：
1. 滤波器系数文件是否存在
2. 采样率是否为 16kHz
3. 音频信号是否为 float32 类型
4. 是否正确应用了群延时补偿
