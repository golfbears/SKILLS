# API 参考文档

## 目录

1. [NumPy API](#numpy-api)
   - [PFBAnalysis](#pfbanalysis)
   - [PFBSynthesis](#pfbsynthesis)
   - [PFBAnalysisBatch](#pfbanalysisbatch)
   - [PFBSynthesisBatch](#pfbsynthesisbatch)

2. [PyTorch API](#pytorch-api)
   - [PFBAnalysisLayer](#pfbanalysplayer)
   - [PFBSynthesisLayer](#pfbsynthesislayer)
   - [PFBTransformLayer](#pfbtransformlayer)

3. [滤波器设计 API](#滤波器设计-api)
   - [KiserFilterDesigner](#kiserfilterdesigner)
   - [compute_kaiser_filter](#compute_kaiser_filter)
   - [compute_polyphase_filters](#compute_polyphase_filters)

---

## NumPy API

### PFBAnalysis

PFB分析器 - 时域到频域的变换

#### 构造函数

```python
PFBAnalysis(
    fft_size: int = 128,
    hop_size: int = 64,
    filter_length: int = 768,
    sample_rate: float = 16000,
    kaiser_beta: float = 12.0,
    cutoff_ratio: float = 0.5,
    preload_filter: Optional[str] = None
)
```

**参数**:
- `fft_size` (int): FFT大小（频率bins数量）。默认128
- `hop_size` (int): 跳跃长度（时间帧间隔）。默认64
- `filter_length` (int): 滤波器长度（通常=fft_size*6）。默认768
- `sample_rate` (float): 采样率(Hz)。默认16000
- `kaiser_beta` (float): Kaiser窗形状参数（越大旁瓣越低）。默认12.0
- `cutoff_ratio` (float): 截止频率比率（相对于Nyquist）。默认0.5
- `preload_filter` (Optional[str]): 预加载的滤波器系数文件路径。默认None

#### 方法

##### process

```python
def process(
    self,
    signal: np.ndarray,
    return_complex: bool = False
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray
```

处理整个信号。

**参数**:
- `signal` (np.ndarray): 输入信号 (samples,)
- `return_complex` (bool): 是否返回复数频谱。默认False

**返回**:
- 如果return_complex=False: `(magnitude, phase)`
  - `magnitude` (np.ndarray): 幅度频谱 (num_frames, fft_size//2)
  - `phase` (np.ndarray): 相位频谱 (num_frames, fft_size//2)
- 如果return_complex=True: `complex_spectrum`
  - `complex_spectrum` (np.ndarray): 复数频谱 (num_frames, fft_size//2)

##### process_frame

```python
def process_frame(
    self,
    input_sample: float
) -> Tuple[np.ndarray, np.ndarray]
```

处理单个采样点，输出当前帧的频谱。

**参数**:
- `input_sample` (float): 输入采样点

**返回**:
- `(magnitude, phase)`: 当前帧的幅度和相位 (fft_size//2,)
- 如果还未完成一帧，返回 `(None, None)`

##### reset

```python
def reset(self) -> None
```

重置内部状态（清空缓冲区）。

##### get_filter_response

```python
def get_filter_response(
    self,
    num_points: int = 1024
) -> Tuple[np.ndarray, np.ndarray]
```

获取原型滤波器的频率响应。

**参数**:
- `num_points` (int): FFT点数。默认1024

**返回**:
- `(freqs, response_db)`: 频率和响应(dB)

---

### PFBSynthesis

PFB合成器 - 频域到时域的变换

#### 构造函数

```python
PFBSynthesis(
    fft_size: int = 128,
    hop_size: int = 64,
    filter_length: int = 768,
    sample_rate: float = 16000,
    kaiser_beta: float = 12.0,
    cutoff_ratio: float = 0.5,
    preload_filter: Optional[str] = None
)
```

参数与 `PFBAnalysis` 相同。

#### 方法

##### process

```python
def process(
    self,
    magnitude: np.ndarray,
    phase: np.ndarray
) -> np.ndarray
```

处理整个频谱序列。

**参数**:
- `magnitude` (np.ndarray): 幅度频谱序列 (num_frames, fft_size//2)
- `phase` (np.ndarray): 相位频谱序列 (num_frames, fft_size//2)

**返回**:
- `reconstructed` (np.ndarray): 重构的时域信号 (samples,)

##### process_complex

```python
def process_complex(
    self,
    spectrum: np.ndarray
) -> np.ndarray
```

直接处理复数频谱。

**参数**:
- `spectrum` (np.ndarray): 复数频谱序列 (num_frames, fft_size//2)

**返回**:
- `reconstructed` (np.ndarray): 重构的时域信号 (samples,)

##### reset

```python
def reset(self) -> None
```

重置内部状态。

---

### PFBAnalysisBatch

批量处理版本的PFB分析器（更快）

继承自 `PFBAnalysis`，构造函数参数相同。

#### 方法

##### process_batch

```python
def process_batch(
    self,
    signal: np.ndarray,
    return_complex: bool = False
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray
```

批量处理信号（比逐帧处理快）。

参数和返回值与 `PFBAnalysis.process` 相同。

---

### PFBSynthesisBatch

批量处理版本的PFB合成器（更快）

继承自 `PFBSynthesis`，构造函数参数相同。

#### 方法

##### process_batch

```python
def process_batch(
    self,
    magnitude: np.ndarray,
    phase: np.ndarray
) -> np.ndarray
```

批量处理（比逐帧处理快）。

参数和返回值与 `PFBSynthesis.process` 相同。

---

## PyTorch API

### PFBAnalysisLayer

PFB分析层 - PyTorch可微分版本

#### 构造函数

```python
PFBAnalysisLayer(
    fft_size: int = 128,
    hop_size: int = 64,
    filter_length: int = 768,
    sample_rate: float = 16000,
    kaiser_beta: float = 12.0,
    cutoff_ratio: float = 0.5,
    learnable: bool = False
)
```

**参数**:
- 与 `PFBAnalysis` 相同的参数
- `learnable` (bool): 是否可学习滤波器系数。默认False

#### 方法

##### forward

```python
def forward(
    self,
    x: torch.Tensor,
    return_complex: bool = True
) -> torch.Tensor
```

前向传播：时域 -> 频域

**参数**:
- `x` (torch.Tensor): 输入音频 (batch, samples)
- `return_complex` (bool): 是否返回复数频谱。默认True

**返回**:
- 如果return_complex=True: 复数频谱 (batch, frequency_bins, time_frames)
- 如果return_complex=False: `(magnitude, phase)`

---

### PFBSynthesisLayer

PFB合成层 - PyTorch可微分版本

#### 构造函数

```python
PFBSynthesisLayer(
    fft_size: int = 128,
    hop_size: int = 64,
    filter_length: int = 768,
    sample_rate: float = 16000,
    kaiser_beta: float = 12.0,
    cutoff_ratio: float = 0.5,
    learnable: bool = False
)
```

参数与 `PFBAnalysisLayer` 相同。

#### 方法

##### forward

```python
def forward(
    self,
    spectrum: torch.Tensor
) -> torch.Tensor
```

前向传播：频域 -> 时域

**参数**:
- `spectrum` (torch.Tensor): 复数频谱 (batch, frequency_bins, time_frames)

**返回**:
- `reconstructed` (torch.Tensor): 重构的时域信号 (batch, samples)

---

### PFBTransformLayer

PFB变换层 - 分析+合成组合

#### 构造函数

```python
PFBTransformLayer(
    fft_size: int = 128,
    hop_size: int = 64,
    filter_length: int = 768,
    sample_rate: float = 16000,
    kaiser_beta: float = 12.0,
    cutoff_ratio: float = 0.5,
    learnable: bool = False
)
```

参数与 `PFBAnalysisLayer` 相同。

#### 方法

##### forward

```python
def forward(
    self,
    x: torch.Tensor
) -> torch.Tensor
```

前向传播：分析 -> 合成（用于验证完美重建）

**参数**:
- `x` (torch.Tensor): 输入音频 (batch, samples)

**返回**:
- `reconstructed` (torch.Tensor): 重构的时域信号 (batch, samples)

##### analysis_forward

```python
def analysis_forward(
    self,
    x: torch.Tensor
) -> torch.Tensor
```

分析前向传播

**参数**:
- `x` (torch.Tensor): 输入音频 (batch, samples)

**返回**:
- `spectrum` (torch.Tensor): 复数频谱 (batch, frequency_bins, time_frames)

##### synthesis_forward

```python
def synthesis_forward(
    self,
    spectrum: torch.Tensor
) -> torch.Tensor
```

合成前向传播

**参数**:
- `spectrum` (torch.Tensor): 复数频谱 (batch, frequency_bins, time_frames)

**返回**:
- `reconstructed` (torch.Tensor): 重构的时域信号 (batch, samples)

---

## 滤波器设计 API

### KiserFilterDesigner

Kaiser窗滤波器设计器

#### 构造函数

```python
KiserFilterDesigner(
    sample_rate: float = 16000,
    cutoff_ratio: float = 0.5,
    ripple_db: float = 60.0,
    transition_width_ratio: float = 0.1
)
```

**参数**:
- `sample_rate` (float): 采样率(Hz)
- `cutoff_ratio` (float): 截止频率比率(相对于Nyquist)
- `ripple_db` (float): 允许的波纹(dB)
- `transition_width_ratio` (float): 过渡带宽度比率

#### 方法

##### design_filter

```python
def design_filter(
    self,
    filter_length: int
) -> np.ndarray
```

设计Kaiser窗FIR滤波器。

**参数**:
- `filter_length` (int): 滤波器长度(必须为奇数)

**返回**:
- `h` (np.ndarray): 滤波器系数

##### analyze_filter

```python
def analyze_filter(
    self,
    h: np.ndarray,
    plot: bool = False
) -> dict
```

分析滤波器特性。

**参数**:
- `h` (np.ndarray): 滤波器系数
- `plot` (bool): 是否绘制频率响应。默认False

**返回**:
- `results` (dict): 滤波器特性字典
  - `filter_length`: 滤波器长度
  - `passband_ripple_db`: 通带波纹(dB)
  - `stopband_attenuation_db`: 阻带衰减(dB)
  - `cutoff_frequency_hz`: 截止频率(Hz)
  - `nyquist_frequency_hz`: Nyquist频率(Hz)

---

### compute_kaiser_filter

```python
def compute_kaiser_filter(
    fft_size: int,
    filter_length: int,
    sample_rate: float = 16000,
    kaiser_beta: float = 12.0,
    cutoff_ratio: float = 0.5
) -> np.ndarray
```

计算用于PFB的Kaiser窗原型滤波器。

**参数**:
- `fft_size` (int): FFT大小
- `filter_length` (int): 滤波器长度(通常=fft_size*6)
- `sample_rate` (float): 采样率
- `kaiser_beta` (float): Kaiser窗beta参数
- `cutoff_ratio` (float): 截止频率比率

**返回**:
- `h` (np.ndarray): 原型滤波器系数 (filter_length,)

---

### compute_polyphase_filters

```python
def compute_polyphase_filters(
    prototype_filter: np.ndarray,
    num_subbands: int
) -> np.ndarray
```

将原型滤波器分解为多相分量。

**参数**:
- `prototype_filter` (np.ndarray): 原型滤波器系数 (L,)
- `num_subbands` (int): 子带数量(=fft_size)

**返回**:
- `polyphase` (np.ndarray): 多相滤波器组 (num_subbands, L//num_subbands)

---

### save_filter_coefficients

```python
def save_filter_coefficients(
    h: np.ndarray,
    config: dict,
    filepath: str
) -> None
```

保存滤波器系数和配置。

**参数**:
- `h` (np.ndarray): 滤波器系数
- `config` (dict): 配置字典
- `filepath` (str): 保存路径

---

### load_filter_coefficients

```python
def load_filter_coefficients(
    filepath: str
) -> Tuple[np.ndarray, dict]
```

加载滤波器系数和配置。

**参数**:
- `filepath` (str): 滤波器系数文件路径

**返回**:
- `(h, config)`: (滤波器系数, 配置字典)

---

### generate_preset_filters

```python
def generate_preset_filters(
    output_dir: str = "./assets/filter_coefficients"
) -> None
```

生成所有预设滤波器。

**参数**:
- `output_dir` (str): 输出目录

**预设配置**:
- `16k_128_64_768`: 16kHz, FFT=128, Hop=64, Filter=768
- `48k_256_128_1536`: 48kHz, FFT=256, Hop=128, Filter=1536
- `16k_64_32_384`: 16kHz, FFT=64, Hop=32, Filter=384

---

## 使用示例

### 示例1：基础NumPy使用

```python
import numpy as np
from scripts.pfb_analysis import PFBAnalysis
from scripts.pfb_synthesis import PFBSynthesis

# 创建分析器和合成器
pfb_analysis = PFBAnalysis(
    fft_size=128,
    hop_size=64,
    filter_length=768,
    sample_rate=16000
)

pfb_synthesis = PFBSynthesis(
    fft_size=128,
    hop_size=64,
    filter_length=768,
    sample_rate=16000
)

# 处理信号
signal = np.random.randn(16000)
magnitude, phase = pfb_analysis.process(signal)
reconstructed = pfb_synthesis.process(magnitude, phase)

# 验证重建质量
error = np.mean((signal[:len(reconstructed)] - reconstructed)**2)
print(f"MSE: {error:.2e}")
```

### 示例2：PyTorch深度学习集成

```python
import torch
from scripts.pfb_pytorch import PFBTransformLayer

# 创建PFB层
pfb_layer = PFBTransformLayer(
    fft_size=128,
    hop_size=64,
    filter_length=768
).cuda()

# 前向传播
audio = torch.randn(2, 16000).cuda()
spectrum = pfb_layer.analysis_forward(audio)
reconstructed = pfb_layer.synthesis_forward(spectrum)

# 完美重建测试
error = torch.mean((audio - reconstructed)**2)
print(f"MSE: {error.item():.2e}")
```

### 示例3：可学习滤波器

```python
import torch
import torch.nn as nn
from scripts.pfb_pytorch import PFBAnalysisLayer, PFBSynthesisLayer

class LearnablePFB(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis = PFBAnalysisLayer(
            fft_size=128,
            hop_size=64,
            filter_length=768,
            learnable=True  # 可学习滤波器
        )
        self.synthesis = PFBSynthesisLayer(
            fft_size=128,
            hop_size=64,
            filter_length=768,
            learnable=True
        )
    
    def forward(self, x):
        spectrum = self.analysis(x)
        reconstructed = self.synthesis(spectrum)
        return reconstructed

# 训练
model = LearnablePFB().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(audio)
    loss = torch.mean((output - audio)**2)  # 完美重建损失
    loss.backward()
    optimizer.step()
```

### 示例4：批量处理

```python
from scripts.pfb_analysis import PFBAnalysisBatch
from scripts.pfb_synthesis import PFBSynthesisBatch

# 批量版本（更快）
pfb_analysis = PFBAnalysisBatch(
    fft_size=128,
    hop_size=64,
    filter_length=768,
    sample_rate=16000
)

pfb_synthesis = PFBSynthesisBatch(
    fft_size=128,
    hop_size=64,
    filter_length=768,
    sample_rate=16000
)

# 批量处理
magnitude, phase = pfb_analysis.process_batch(signal)
reconstructed = pfb_synthesis.process_batch(magnitude, phase)
```

### 示例5：自定义滤波器设计

```python
from scripts.filter_design import KiserFilterDesigner, compute_kaiser_filter

# 方法1：使用设计器
designer = KiserFilterDesigner(
    sample_rate=48000,
    cutoff_ratio=0.5,
    ripple_db=80.0  # 更高的阻带衰减
)

h = designer.design_filter(filter_length=1536)
results = designer.analyze_filter(h, plot=True)

# 方法2：直接计算
h = compute_kaiser_filter(
    fft_size=256,
    filter_length=1536,
    sample_rate=48000,
    kaiser_beta=15.0  # 更高的beta
)

# 保存滤波器
from scripts.filter_design import save_filter_coefficients
config = {
    'fft_size': 256,
    'hop_size': 128,
    'filter_length': 1536,
    'sample_rate': 48000
}
save_filter_coefficients(h, config, 'custom_filter.npy')

# 在PFB中使用预加载的滤波器
pfb = PFBAnalysis(
    fft_size=256,
    hop_size=128,
    filter_length=1536,
    sample_rate=48000,
    preload_filter='custom_filter.npy'
)
```
