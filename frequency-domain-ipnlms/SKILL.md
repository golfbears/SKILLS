# Frequency Domain IPNLMS AEC Skill

## 概述

基于Athena-signal原生C实现的频域IPNLMS（Improved Proportionate Normalized Least Mean Square）自适应滤波器。本Skill提供两个版本：

- **PyTorch版本** (`ipnlms_aec.py`): 适合深度学习集成
- **NumPy版本** (`numpy_ipnlms_aec.py`): 纯Python实现，无需PyTorch依赖

两个版本都支持真正的多块历史帧处理和在线自适应更新，参数与Athena C模型对齐。

## 算法原理

### 频域块自适应滤波器结构
- **FFT帧长**: 256采样（可配置）
- **帧移**: FFT帧长/2（50%重叠）
- **有效频点**: FFT帧长/2+1（如256→129）
- **滤波器块数**: N（可配置4~8块，默认8）
- **滤波器阶数**: 每个频点有N个复数系数，每个系数对应一个历史块

### 核心公式

**回声估计**（根据Athena C代码）:
```
每频点f的估计回声:
  echo_est[f] = Σ conj(coef[f,b]) × ref_history[f,b]  (b=0~N-1)
  
其中:
  conj(coef) = coef_real - j × coef_imag
  ref_history: 该频点最近N帧参考信号的FFT值
```

**误差信号**:
```
error[f] = mic_fft[f] - echo_est[f]
```

**IPNLMS系数更新**（真正的比例化步长）:
```
Step 1: 计算IPNLMS比例因子 kl
  kl[f,b] = (1-α)/(2N) + (1+α) × |coef[f,b]|² / (Σ|coef[f]|² + ε)
  
  - α ∈ [-1, 1]，通常取0.5
  - kl越大，该系数更新越快（比例化步长）
  - α=1时退化为PNLMS，α=-1时退化为NLMS

Step 2: 计算参考信号功率
  power[f] = Σ |ref_history[f,b]|²
  
Step 3: 计算归一化步长
  μ_normalized = μ / (power + β)
  
Step 4: 计算 ref × conj(error)
  product = ref_history × conj(error)
  
Step 5: 应用IPNLMS更新
  coef[f,b] += μ_normalized[f] × kl[f,b] × product[f,b]
```

### 稳定性措施

为保证算法在实际场景中的稳定性，实现了以下措施：

1. **动态正则化**: `reg = β × (power.max() + 1)`
2. **更新幅度限制**: `clamp(update, -0.01, 0.01)`
3. **系数幅度限制**: `max_coef = 2.0`
4. **功率平滑估计**: `power = λ × power_old + (1-λ) × power_new`

## 使用方法

### NumPy版本（推荐，无需PyTorch）

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("C:/Users/charles.wu/.codebuddy/skills/frequency-domain-ipnlms")))

from numpy_ipnlms_aec import NumPyIPNLMS

# 初始化（与Athena C模型对齐的参数）
aec = NumPyIPNLMS(
    fft_size=256,      # FFT点数
    num_blocks=8,      # 滤波器块数
    mu=0.5,            # 步长因子（与C模型对齐）
    alpha=0.5,         # IPNLMS alpha参数
    beta=1e-8          # 正则化因子（与C模型对齐）
)

# 处理频谱
# mic_spectrum: (T, F) complex64 - 麦克风频谱
# ref_spectrum: (T, F) complex64 - 远端参考频谱
error_spectrum, echo_estimate = aec.process(mic_spectrum, ref_spectrum)

# 重置滤波器状态（处理新文件时需要）
aec.reset()
```

### PyTorch版本

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("C:/Users/charles.wu/.codebuddy/skills/frequency-domain-ipnlms")))

from ipnlms_aec import FrequencyDomainIPNLMS

# 初始化（推荐参数）
aec = FrequencyDomainIPNLMS(
    fft_size=256,      # FFT点数
    num_blocks=8,      # 滤波器块数
    mu=0.5,            # 步长因子（与C模型对齐）
    alpha=0.5,         # IPNLMS alpha参数
    beta=1e-8          # 正则化因子（与C模型对齐）
)

# 前向传播
# mic_fft: (B, T, F) complex64 - 麦克风频谱
# ref_fft: (B, T, F) complex64 - 远端参考频谱
error_fft, echo_estimate = aec(mic_fft, ref_fft)

# 重置滤波器状态（处理新文件时需要）
aec.reset()
```

### 完整处理流程（配合PFB时频变换）

```python
import torch
import soundfile as sf

# 导入IPNLMS skill
from ipnlms_aec import FrequencyDomainIPNLMS

# 导入PFB skill（如果有的话）
from pfb_analyze_synthesize import PFB

# 读取音频
mic_audio, sr = sf.read("mic.wav")      # (samples,)
ref_audio, _ = sf.read("ref.wav")       # (samples,)

# 转为tensor
mic_tensor = torch.from_numpy(mic_audio).float().unsqueeze(0)  # (1, samples)
ref_tensor = torch.from_numpy(ref_audio).float().unsqueeze(0)  # (1, samples)

# 时频变换（使用PFB或STFT）
pfb = PFB(fft_size=256)
mic_fft = pfb.analyze(mic_tensor)  # (1, T, 129) complex64
ref_fft = pfb.analyze(ref_tensor)  # (1, T, 129) complex64

# 初始化IPNLMS
aec = FrequencyDomainIPNLMS(fft_size=256, num_blocks=8, mu=0.1, alpha=0.5)
aec.train()  # 启用训练模式进行自适应更新

# 回声消除
error_fft, echo_fft = aec(mic_fft, ref_fft)

# 时域重建
error_audio = pfb.synthesize(error_fft)  # (1, samples)

# 保存结果
sf.write("aec_error.wav", error_audio.squeeze().numpy(), sr)
```

## 参数说明

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| fft_size | 256 | 256, 512 | FFT点数，影响频率分辨率 |
| num_blocks | 8 | 4~8 | 滤波器块数，对应时域延迟长度 |
| mu | 0.5 | 0.1~1.0 | 步长因子（与Athena C模型对齐） |
| alpha | 0.5 | 0.0~0.9 | IPNLMS比例因子，0.5平衡收敛速度和稳定性 |
| beta | 1e-8 | 1e-6~1e-10 | 正则化因子（与Athena C模型对齐） |

### 参数调优建议

- **小腔体场景（手机/耳机）**: `num_blocks=6~8`, `mu=0.1`, `alpha=0.5`
- **快速收敛场景**: `num_blocks=4`, `mu=0.2`, `alpha=0.7`
- **稳定性优先**: `num_blocks=8`, `mu=0.05`, `alpha=0.3`, `beta=1e-2`

## 性能指标

- **参数规模**: 129 bins × 8 blocks × 2 (复数实部虚部) ≈ 2K 参数
- **计算复杂度**: O(F × N) 每帧，约2K次乘加运算
- **内存占用**: 约4KB（系数）+ 约8KB（历史缓冲区）
- **收敛时间**: 约0.5~2秒（取决于场景）
- **适用场景**: 端侧AEC，小腔体（手机/耳机/头戴设备）

## 实测结果

### 测试环境
- 音频长度: 261秒
- 采样率: 16kHz
- FFT大小: 256
- 滤波器块数: 8

### 性能表现
- 功率降低: ~7.4~7.8 dB
- 收敛稳定: 无发散问题
- 处理速度: 实时处理

### 不同参数对比

| 参数组合 | 功率降低 | 稳定性 |
|----------|----------|--------|
| mu=0.1, alpha=0.5, blocks=8 | 7.8 dB | 优秀 |
| mu=0.05, alpha=0.5, blocks=8 | 7.5 dB | 优秀 |
| mu=0.1, alpha=0.7, blocks=6 | 7.4 dB | 良好 |

## 算法特点

### 优点
1. **真正的IPNLMS**: 每个系数有独立的比例化步长，加速稀疏系统收敛
2. **频域处理**: 计算效率高，易于与神经网络集成
3. **因果性保证**: 仅使用历史帧，适合实时处理
4. **参数可控**: 可根据场景调整块数、步长等参数
5. **稳定性好**: 内置多种稳定性保障措施

### 限制
1. **需要远端参考信号**: 要求ref信号可用
2. **频谱污染**: 频域处理可能引入轻微频谱失真
3. **参数敏感**: 需要根据场景调优参数

## 与PFB Skill配合使用

本Skill设计为与时频变换Skill（如PFB）配合使用：

```
时域信号 → [PFB Analyze] → FFT频谱 → [IPNLMS AEC] → 误差频谱 → [PFB Synthesize] → 时域输出
              (1,T,129)                (1,T,129)                              (1,T,129)
```

## 参考实现

本实现基于Athena-signal原生C代码，参考文件：
- `D:/simulation/athena-signal-master-pure-c/athena_signal/kernels/dios_ssp_aec/dios_ssp_aec_firfilter.c`
- `D:/simulation/athena-signal-master-pure-c/athena_signal/kernels/dios_ssp_aec/dios_ssp_aec_pbfdaf.c`

关键实现细节遵循C代码的逻辑，特别是：
- 回声估计公式: `y = conj(h) * x`
- IPNLMS比例因子计算
- 系数更新公式
