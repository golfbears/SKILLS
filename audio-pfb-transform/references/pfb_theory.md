# PFB (Polyphase Filter Bank) 理论基础

## 1. 概述

PFB（多相滤波器组）是一种高效的时频变换技术，它在保持完美重建能力的同时，提供了比传统STFT更优的频率分辨率和更小的旁瓣泄漏。

## 2. 数学原理

### 2.1 离散时间信号分析

给定离散时间信号 \(x[n]\)，我们希望将其分解为多个子带信号。设分析滤波器组有 \(M\) 个子带，第 \(k\) 个子带的输出为：

\[ y_k[m] = \sum_{n} h_k[n] x[nM - m] \]

其中 \(h_k[n]\) 是第 \(k\) 个分析滤波器。

### 2.2 多相分解

原型低通滤波器 \(h[n]\) 可以分解为 \(M\) 个多相分量：

\[ h_k[n] = h[k + nM], \quad k = 0, 1, ..., M-1 \]

这种分解使得计算效率显著提高，因为每个子带只需要处理 \(1/M\) 的数据。

### 2.3 分析滤波器组

分析滤波器组的输出可以通过多相形式表示为：

\[ Y_k(z) = \sum_{i=0}^{M-1} H_i(z^M) z^{-i} X(z) W^{ik} \]

其中：
- \(H_i(z)\) 是第 \(i\) 个多相分量
- \(W = e^{-j2\pi/M}\) 是单位根
- \(X(z)\) 是输入信号的Z变换

### 2.4 合成滤波器组

合成滤波器组将子带信号重建为原始信号：

\[ \hat{x}[n] = \sum_{k} g_k[n] * (y_k \uparrow M)[n] \]

其中 \(g_k[n]\) 是合成滤波器，\(\uparrow M\) 表示M倍上采样。

### 2.5 完美重建条件

为了实现完美重建，分析滤波器 \(h_k[n]\) 和合成滤波器 \(g_k[n]\) 必须满足：

\[ \sum_{k} H_k(z) G_k(z) = c z^{-d} \]

其中 \(c\) 是常数，\(d\) 是延时。当满足此条件时，重构信号满足：

\[ \hat{x}[n] = c x[n-d] \]

## 3. 与STFT的关系

### 3.1 STFT的本质

STFT可以看作是一种特殊的滤波器组，其窗函数 \(w[n]\) 用作低通滤波器：

\[ STFT\{x[n]\}[m,k] = \sum_{n} x[n] w[n-mR] e^{-j2\pi kn/N} \]

其中 \(R\) 是跳跃长度（hop size），\(N\) 是FFT大小。

### 3.2 PFB相对于STFT的优势

| 特性 | STFT | PFB |
|------|------|-----|
| 旁瓣水平 | 窗函数限制（-13dB 到 -60dB） | 可设计到<-100dB |
| 频率泄漏 | 中等 | 极低 |
| 完美重建 | 需50%重叠 | 天然支持 |
| 计算效率 | 高 | 略低但可通过优化 |
| 灵活性 | 受窗函数限制 | 完全可设计 |

### 3.3 联系

当PFB的滤波器长度等于FFT大小时，PFB退化为与STFT等价的形式。因此，PFB可以看作是STFT的泛化和扩展。

## 4. 滤波器设计

### 4.1 Kaiser窗滤波器

Kaiser窗是一种可调窗函数，通过参数 \(\beta\) 控制旁瓣水平：

\[ w[n] = \frac{I_0(\beta \sqrt{1 - (2n/(N-1) - 1)^2})}{I_0(\beta)} \]

其中 \(I_0\) 是修正的零阶贝塞尔函数。

### 4.2 原型滤波器设计

原型低通滤波器的设计步骤：

1. 确定截止频率 \(\omega_c\) 和过渡带宽 \(\Delta\omega\)
2. 计算所需的滤波器长度 \(L\)：
   \[ L \approx \frac{A_s - 7.95}{14.36 \Delta\omega / (2\pi)} \]
   其中 \(A_s\) 是期望的阻带衰减(dB)
3. 生成理想低通脉冲响应：
   \[ h_d[n] = \frac{\sin(\omega_c (n - L/2))}{\pi (n - L/2)} \]
4. 应用Kaiser窗：
   \[ h[n] = h_d[n] w[n] \]

### 4.3 实际参数选择

对于音频处理，典型参数为：

- **16kHz采样率**：
  - FFT大小：128
  - 跳跃长度：64
  - 滤波器长度：768（6倍FFT大小）
  - Kaiser beta：12.0

- **48kHz采样率**：
  - FFT大小：256
  - 跳跃长度：128
  - 滤波器长度：1536
  - Kaiser beta：12.0

## 5. 实现考虑

### 5.1 循环缓冲区

为了高效处理流式数据，使用循环缓冲区：

```python
# 写入索引
buffer[index] = input_sample
index = (index + 1) % buffer_length

# 读取最新数据
buffer_ordered = buffer[index:] + buffer[:index]
```

### 5.2 多相滤波的矩阵表示

多相滤波可以表示为矩阵乘法：

\[ \mathbf{y} = \mathbf{H} \mathbf{x} \]

其中：
- \(\mathbf{H}\) 是多相滤波矩阵
- \(\mathbf{x}\) 是输入信号向量
- \(\mathbf{y}\) 是滤波后输出

这种表示允许使用BLAS优化的矩阵运算。

### 5.3 FFT加速

对于较大的子带数量（M > 16），可以使用FFT加速滤波：

1. 构造多相滤波器的DFT
2. 通过频域卷积代替时域卷积
3. 利用FFT的 \(O(N \log N)\) 复杂度

## 6. 应用场景

### 6.1 回声消除(AEC)

PFB在AEC中的优势：
- 高频率分辨率提高回声路径建模精度
- 低旁瓣减少频域泄漏的影响
- 完美重建避免引入失真

### 6.2 语音增强

PFB用于语音增强：
- 语音和噪声在PFB域可分离性更好
- 相位估计更准确
- 音乐噪声更少

### 6.3 音频编码

PFB在音频编码中的应用：
- MP3、AAC等格式使用PFB作为分析/合成滤波器
- 高质量音频编码的标准选择
- 支持可变比特率编码

## 7. 性能优化

### 7.1 SIMD优化

利用SIMD指令（如AVX）加速多相滤波：

```c
// AVX实现示例
__m256 sum = _mm256_setzero_ps();
for (int i = 0; i < length; i += 8) {
    __m256 data = _mm256_loadu_ps(input + i);
    __m256 coeff = _mm256_loadu_ps(filter + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(data, coeff));
}
```

### 7.2 GPU加速

使用CUDA或PyTorch GPU实现：
- 并行处理多个子带
- 批量处理多个信号
- 利用共享内存优化访问模式

### 7.3 内存预分配

预分配所有临时缓冲区，避免运行时分配：

```python
# 初始化时预分配
self.circ_buffer = np.zeros(filter_length)
self.output_buffer = np.zeros(filter_length)
self.temp_fft = np.zeros(fft_size, dtype=np.complex128)
```

## 8. 验证和测试

### 8.1 完美重建测试

```python
# 分析-合成
magnitude, phase = analysis(signal)
reconstructed = synthesis(magnitude, phase)

# 计算SNR
error = signal[:len(reconstructed)] - reconstructed
snr = 10 * log10(mean(signal**2) / mean(error**2))
assert snr > 60 dB  # 完美重建阈值
```

### 8.2 频率响应测试

验证滤波器频率响应符合设计规格：
- 通带波纹 < 0.1 dB
- 阻带衰减 > 60 dB
- 过渡带宽符合设计

### 8.3 因果性测试

确保系统是因果的（不使用未来信息）：

```python
# 测试1：相同输入产生相同输出
output1 = process(signal)
output2 = process(signal)
assert allclose(output1, output2)

# 测试2：逐帧处理等同于批量处理
frame_output = [process_frame(f) for f in frames]
batch_output = process_batch(frames)
assert allclose(concat(frame_output), batch_output)
```

## 9. 参考文献

1. Vaidyanathan, P. P. (1993). *Multirate Systems and Filter Banks*. Prentice Hall.
2. Fliege, N. J. (1994). *Multirate Digital Signal Processing*. Wiley.
3. Bellanger, M. G. (2000). *Digital Processing of Signals: Theory and Practice*. Wiley.
4. Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
5. Strang, G., & Nguyen, T. (1996). *Wavelets and Filter Banks*. Wellesley-Cambridge Press.

## 10. 扩展阅读

- **M通道滤波器组**：PFB的一般化
- **小波变换**：与PFB的关系
- **自适应滤波器组**：根据信号特性调整
- **非均匀滤波器组**：人耳听觉模型
