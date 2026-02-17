# Athena C代码解析

## 概述

Athena是一个开源的声学回声消除(AEC)项目，其核心使用了PFB作为时频变换前端。本文档解析Athena C代码中的PFB实现逻辑。

## 关键数据结构

### 滤波器系数存储

```c
// 原型滤波器系数（768点）
static float h_proto[768];

// 多相滤波器组（128 x 6）
static float h_polyphase[128][6];

// 循环缓冲区
static float circ_buffer[768];
static int buffer_index = 0;
```

### PFB状态

```c
typedef struct {
    int fft_size;          // 128
    int hop_size;          // 64
    int filter_length;     // 768
    
    float *circ_buffer;    // 循环缓冲区
    int buffer_index;      // 当前写入位置
    
    float *polyphase_filters;  // 多相滤波器
    
    fftwf_plan fft_plan;   // FFTW计划
} PFBAnalysis;
```

## 分析器实现

### 初始化

```c
void pfb_analysis_init(PFBAnalysis *pfb, int fft_size, int hop_size, int filter_length) {
    pfb->fft_size = fft_size;
    pfb->hop_size = hop_size;
    pfb->filter_length = filter_length;
    
    // 分配缓冲区
    pfb->circ_buffer = (float*)calloc(filter_length, sizeof(float));
    pfb->buffer_index = 0;
    
    // 分配多相滤波器
    pfb->polyphase_filters = (float*)malloc(fft_size * (filter_length / fft_size) * sizeof(float));
    
    // 设计并填充多相滤波器
    design_polyphase_filters(pfb->polyphase_filters, filter_length, fft_size);
    
    // 初始化FFTW计划
    pfb->fft_plan = fftwf_plan_dft_1d(fft_size, NULL, NULL, FFTW_FORWARD, FFTW_MEASURE);
}
```

### 滤波器设计

```c
void design_kaiser_filter(float *h, int length, float beta, float cutoff) {
    // 生成时间序列
    for (int i = 0; i < length; i++) {
        float t = (float)i - (length - 1) / 2.0f;
        t = t / (float)length;
        
        // 理想低通
        float h_ideal = 2.0f * cutoff * sincf(2.0f * cutoff * t);
        
        // Kaiser窗
        float alpha = beta * 3.141592653589793f;
        float arg = 1.0f - (4.0f * t * t);
        float window = (arg >= 0) ? bessel_i0(alpha * sqrtf(arg)) / bessel_i0(alpha) : 0.0f;
        
        // 组合
        h[i] = h_ideal * window;
    }
    
    // 归一化
    float sum = 0;
    for (int i = 0; i < length; i++) {
        sum += fabsf(h[i]);
    }
    for (int i = 0; i < length; i++) {
        h[i] /= sum;
    }
}

void design_polyphase_filters(float *polyphase, int filter_length, int num_subbands) {
    // 生成原型滤波器
    float *h_proto = (float*)malloc(filter_length * sizeof(float));
    design_kaiser_filter(h_proto, filter_length, 12.0f, 0.5f);
    
    // 分解为多相分量
    for (int i = 0; i < num_subbands; i++) {
        for (int j = 0; i + j * num_subbands < filter_length; j++) {
            int idx = i * (filter_length / num_subbands) + j;
            polyphase[idx] = h_proto[i + j * num_subbands];
        }
    }
    
    free(h_proto);
}
```

### 逐帧处理

```c
void pfb_analysis_process_frame(PFBAnalysis *pfb, float input, float *magnitude, float *phase) {
    // 写入循环缓冲区
    pfb->circ_buffer[pfb->buffer_index] = input;
    pfb->buffer_index = (pfb->buffer_index + 1) % pfb->filter_length;
    
    // 检查是否到达输出帧
    static int frame_counter = 0;
    if (frame_counter % pfb->hop_size != 0) {
        frame_counter++;
        return;
    }
    
    // 重排序缓冲区以获取最新的filter_length个采样
    float buffer_ordered[768];
    for (int i = 0; i < pfb->filter_length; i++) {
        buffer_ordered[i] = pfb->circ_buffer[(pfb->buffer_index + i) % pfb->filter_length];
    }
    
    // 多相滤波
    fftwf_complex filtered[pfb->fft_size];
    for (int i = 0; i < pfb->fft_size; i++) {
        float sum_real = 0, sum_imag = 0;
        
        // 获取多相分量（降采样）
        float polyphase[6];  // filter_length / fft_size = 6
        for (int j = 0; i + j * pfb->fft_size < pfb->filter_length; j++) {
            polyphase[j] = buffer_ordered[i + j * pfb->fft_size];
        }
        
        // 卷积
        for (int j = 0; j < 6; j++) {
            sum_real += polyphase[j] * pfb->polyphase_filters[i * 6 + j];
        }
        
        filtered[i] = sum_real + 0.0f * I;
    }
    
    // FFT
    fftwf_execute_dft(pfb->fft_plan, filtered, filtered);
    
    // 输出幅度和相位（只取正频率部分）
    for (int i = 0; i < pfb->fft_size / 2; i++) {
        magnitude[i] = sqrtf(filtered[i][0] * filtered[i][0] + filtered[i][1] * filtered[i][1]);
        phase[i] = atan2f(filtered[i][1], filtered[i][0]);
    }
    
    frame_counter++;
}
```

## 合成器实现

### 初始化

```c
void pfb_synthesis_init(PFBSynthesis *pfb, int fft_size, int hop_size, int filter_length) {
    pfb->fft_size = fft_size;
    pfb->hop_size = hop_size;
    pfb->filter_length = filter_length;
    
    // 分配输出缓冲区（用于Overlap-Add）
    pfb->output_buffer = (float*)calloc(filter_length, sizeof(float));
    
    // 使用相同的滤波器（与分析器一致）
    pfb->polyphase_filters = (float*)malloc(fft_size * (filter_length / fft_size) * sizeof(float));
    design_polyphase_filters(pfb->polyphase_filters, filter_length, fft_size);
    
    // 初始化IFFTW计划
    pfb->ifft_plan = fftwf_plan_dft_1d(fft_size, NULL, NULL, FFTW_BACKWARD, FFTW_MEASURE);
    
    // 时延补偿
    pfb->delay = (filter_length - 1) / 2;
}
```

### 逐帧处理

```c
void pfb_synthesis_process_frame(PFBSynthesis *pfb, float *magnitude, float *phase, float *output) {
    // 重构复数频谱
    fftwf_complex spectrum[pfb->fft_size];
    for (int i = 0; i < pfb->fft_size / 2; i++) {
        spectrum[i][0] = magnitude[i] * cosf(phase[i]);
        spectrum[i][1] = magnitude[i] * sinf(phase[i]);
    }
    // 负频率（共轭对称）
    for (int i = 1; i < pfb->fft_size / 2; i++) {
        spectrum[pfb->fft_size - i][0] = spectrum[i][0];
        spectrum[pfb->fft_size - i][1] = -spectrum[i][1];
    }
    spectrum[0][1] = 0;
    spectrum[pfb->fft_size / 2][1] = 0;
    
    // IFFT
    fftwf_complex time_domain[pfb->fft_size];
    fftwf_execute_dft(pfb->ifft_plan, spectrum, time_domain);
    
    // 多相合成
    float synthesized[pfb->filter_length];
    memset(synthesized, 0, pfb->filter_length * sizeof(float));
    
    for (int i = 0; i < pfb->fft_size; i++) {
        float time_sample = time_domain[i][0];  // 实部
        
        // 应用多相滤波器
        for (int j = 0; i + j * pfb->fft_size < pfb->filter_length; j++) {
            int idx = i + j * pfb->fft_size;
            synthesized[idx] += time_sample * pfb->polyphase_filters[i * 6 + j];
        }
    }
    
    // Overlap-Add到输出缓冲区
    for (int i = 0; i < pfb->filter_length; i++) {
        pfb->output_buffer[i] += synthesized[i];
    }
    
    // 输出hop_size个采样
    for (int i = 0; i < pfb->hop_size; i++) {
        output[i] = pfb->output_buffer[i];
    }
    
    // 缓冲区移位
    for (int i = 0; i < pfb->filter_length - pfb->hop_size; i++) {
        pfb->output_buffer[i] = pfb->output_buffer[i + pfb->hop_size];
    }
    memset(pfb->output_buffer + pfb->filter_length - pfb->hop_size, 0, pfb->hop_size * sizeof(float));
}
```

## 关键算法解析

### 1. 循环缓冲区访问

循环缓冲区的高效访问：

```c
// 写入
buffer[index] = value;
index = (index + 1) % length;

// 读取（从index开始的length个元素）
for (int i = 0; i < length; i++) {
    ordered[i] = buffer[(index + i) % length];
}
```

### 2. 多相卷积

多相卷积的本质是降采样后卷积：

```c
// 朴素卷积（慢）
for (int n = 0; n < N; n++) {
    y[n] = 0;
    for (int k = 0; k < L; k++) {
        if (n - k >= 0) {
            y[n] += x[n-k] * h[k];
        }
    }
}

// 多相卷积（快）
for (int i = 0; i < M; i++) {  // M个子带
    y[i] = 0;
    for (int j = 0; j < L/M; j++) {
        y[i] += x[i + j*M] * h_polyphase[i][j];
    }
}
```

### 3. 完美重建验证

```c
float test_perfect_reconstruction(PFBAnalysis *analysis, PFBSynthesis *synthesis, 
                                   float *input, int length) {
    // 重置状态
    memset(analysis->circ_buffer, 0, analysis->filter_length * sizeof(float));
    analysis->buffer_index = 0;
    memset(synthesis->output_buffer, 0, synthesis->filter_length * sizeof(float));
    
    float *reconstructed = (float*)malloc(length * sizeof(float));
    int output_index = 0;
    
    // 处理每个采样
    for (int i = 0; i < length; i++) {
        float mag[64], ph[64];
        pfb_analysis_process_frame(analysis, input[i], mag, ph);
        
        float frame_output[64];
        pfb_synthesis_process_frame(synthesis, mag, ph, frame_output);
        
        for (int j = 0; j < 64 && output_index < length; j++) {
            reconstructed[output_index++] = frame_output[j];
        }
    }
    
    // 计算误差
    float error_sum = 0;
    for (int i = 0; i < length; i++) {
        float error = input[i] - reconstructed[i];
        error_sum += error * error;
    }
    
    float mse = error_sum / length;
    float snr = 10.0f * log10f(mse);
    
    free(reconstructed);
    return snr;
}
```

## 性能优化技巧

### 1. FFTW计划缓存

```c
// 避免每次都重新创建计划
static fftwf_plan cached_plan = NULL;

fftwf_plan get_fft_plan(int size) {
    if (cached_plan == NULL) {
        float *in = fftwf_alloc_complex(size);
        float *out = fftwf_alloc_complex(size);
        cached_plan = fftwf_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_MEASURE);
        fftwf_free(in);
        fftwf_free(out);
    }
    return cached_plan;
}
```

### 2. SIMD向量化

```c
// 使用AVX加速内积计算
#include <immintrin.h>

float dot_product_avx(float *a, float *b, int length) {
    __m256 sum = _mm256_setzero_ps();
    
    for (int i = 0; i < length; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    
    // 水平求和
    float result[8];
    _mm256_storeu_ps(result, sum);
    return result[0] + result[1] + result[2] + result[3] + 
           result[4] + result[5] + result[6] + result[7];
}
```

### 3. 内存对齐

```c
// 确保缓冲区对齐到32字节边界（AVX要求）
float *circ_buffer = (float*)aligned_alloc(32, filter_length * sizeof(float));

// 或使用FFTW的分配器
float *circ_buffer = (float*)fftwf_malloc(filter_length * sizeof(float));
```

## Python移植要点

将C代码移植到Python时需要注意：

1. **循环缓冲区**：使用 `np.roll` 或索引计算
2. **多相访问**：使用切片 `x[i::M]`
3. **FFT**：使用 `scipy.fft.fft` 而不是 `np.fft`
4. **性能**：考虑使用Numba或Cython加速
5. **复数处理**：Python的复数类型是原生的

```python
# Python多相滤波示例
def polyphase_filter(buffer, polyphase_filters):
    filtered = np.zeros(fft_size, dtype=np.complex128)
    
    for i in range(fft_size):
        # 多相分量（降采样）
        polyphase_component = buffer[i::fft_size]
        
        # 卷积
        filtered[i] = np.sum(polyphase_component * polyphase_filters[i])
    
    return filtered
```

## 总结

Athena的PFB实现展示了：
- 高效的循环缓冲区管理
- 多相分解优化计算效率
- FFTW进行快速FFT
- Overlap-Add实现完美重建

我们的Python实现基于相同的原理，同时利用NumPy/PyTorch的向量化操作实现高性能。
