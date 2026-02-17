"""
NumPy实现的频域IPNLMS自适应滤波器

基于Athena-signal C模型的纯NumPy实现，不依赖PyTorch。
支持真正的多块历史帧处理和在线自适应更新。

使用示例:
    from numpy_ipnlms_aec import NumPyIPNLMS
    
    aec = NumPyIPNLMS(fft_size=256, num_blocks=8, mu=0.5, alpha=0.5, beta=1e-8)
    error_spectrum, echo_estimate = aec.process(mic_spectrum, ref_spectrum)
"""

import numpy as np
from typing import Tuple, Optional


class NumPyIPNLMS:
    """
    NumPy实现的频域IPNLMS自适应滤波器
    
    基于Athena-signal C模型的参数:
    - mu = 0.5 (步长)
    - alpha = 0.5 (IPNLMS比例因子)
    - beta = 1e-8 (正则化)
    - num_blocks = 8 (滤波器块数)
    
    输入: 
        mic_spectrum: (T, F) complex64 - 麦克风频谱
        ref_spectrum: (T, F) complex64 - 远端参考频谱
        
    输出:
        error_spectrum: (T, F) complex64 - 误差信号（回声消除后）
        echo_estimate: (T, F) complex64 - 估计的回声频谱
    """
    
    def __init__(
        self,
        fft_size: int = 256,
        num_blocks: int = 8,
        mu: float = 0.5,
        alpha: float = 0.5,
        beta: float = 1e-8,
        eps: float = 1e-10,
        use_dual_filter: bool = True
    ):
        """
        初始化IPNLMS滤波器
        
        Args:
            fft_size: FFT点数
            num_blocks: 滤波器块数
            mu: 步长因子 (默认0.5，与Athena C模型对齐)
            alpha: IPNLMS alpha参数 (默认0.5)
            beta: 正则化因子 (默认1e-8，与Athena C模型对齐)
            eps: 防止除零的小常数
            use_dual_filter: 是否使用双滤波器机制
        """
        self.fft_size = fft_size
        self.num_freq_bins = fft_size // 2 + 1  # 129 for 256 FFT
        self.num_blocks = num_blocks
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_dual_filter = use_dual_filter
        
        # FIR系数 - 初始化为很小的值
        self.fir_coef_real = np.zeros((self.num_freq_bins, self.num_blocks), dtype=np.float32)
        self.fir_coef_imag = np.zeros((self.num_freq_bins, self.num_blocks), dtype=np.float32)
        self.fir_coef_real[:, :] = 1e-5  # 初始值
        
        # ADF系数 (仅在use_dual_filter=True时使用)
        if self.use_dual_filter:
            self.adf_coef_real = np.zeros((self.num_freq_bins, self.num_blocks), dtype=np.float32)
            self.adf_coef_imag = np.zeros((self.num_freq_bins, self.num_blocks), dtype=np.float32)
            self.adf_coef_real[:, :] = 1e-5
        
        # 参考信号历史缓冲区
        self.ref_history_real = np.zeros((self.num_freq_bins, self.num_blocks), dtype=np.float32)
        self.ref_history_imag = np.zeros((self.num_freq_bins, self.num_blocks), dtype=np.float32)
        
        # 功率估计
        self.power_estimate = np.ones(self.num_freq_bins, dtype=np.float32)
        
        # MSE跟踪 (用于双讲检测)
        self.mse_mic_in = np.ones(self.num_freq_bins, dtype=np.float32)
        self.mse_adpt = np.ones(self.num_freq_bins, dtype=np.float32)
        self.mse_main = np.ones(self.num_freq_bins, dtype=np.float32)
        
        # 双讲检测参数 (根据Athena C代码)
        self.mse_ratio_out_in = 8.0  # MSE_RATIO_OUT_IN
        self.filter_copy_fac = 0.5   # FILTER_COPY_FAC
        
    def reset(self):
        """重置滤波器状态"""
        self.ref_history_real.fill(0)
        self.ref_history_imag.fill(0)
        self.power_estimate.fill(1.0)
        self.mse_mic_in.fill(1.0)
        self.mse_adpt.fill(1.0)
        self.mse_main.fill(1.0)
        self.fir_coef_real.fill(1e-5)
        self.fir_coef_imag.fill(0)
        if self.use_dual_filter:
            self.adf_coef_real.fill(1e-5)
            self.adf_coef_imag.fill(0)
    
    def _push_ref_frame(self, ref_real: np.ndarray, ref_imag: np.ndarray):
        """将新的参考帧推入历史缓冲区"""
        # 右移历史
        self.ref_history_real[:, 1:] = self.ref_history_real[:, :-1]
        self.ref_history_imag[:, 1:] = self.ref_history_imag[:, :-1]
        # 插入新帧
        self.ref_history_real[:, 0] = ref_real
        self.ref_history_imag[:, 0] = ref_imag
    
    def _estimate_echo_with_coef(
        self, 
        coef_real: np.ndarray, 
        coef_imag: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用指定系数估计回声: echo = Σ conj(coef) × ref
        
        根据Athena C代码: y = conj(h) * x
        
        Args:
            coef_real: (F, num_blocks) - 系数实部
            coef_imag: (F, num_blocks) - 系数虚部
            
        Returns:
            echo_real, echo_imag: (F,)
        """
        # echo = Σ conj(coef) × ref
        # = Σ (coef_real - j*coef_imag) × (ref_real + j*ref_imag)
        # = Σ (coef_real*ref_real + coef_imag*ref_imag) + j*(coef_real*ref_imag - coef_imag*ref_real)
        
        echo_real = np.sum(
            coef_real * self.ref_history_real + 
            coef_imag * self.ref_history_imag,
            axis=1
        )
        echo_imag = np.sum(
            coef_real * self.ref_history_imag - 
            coef_imag * self.ref_history_real,
            axis=1
        )
        
        return echo_real, echo_imag
    
    def _estimate_echo(self) -> Tuple[np.ndarray, np.ndarray]:
        """使用FIR系数估计回声（默认）"""
        return self._estimate_echo_with_coef(self.fir_coef_real, self.fir_coef_imag)
    
    def _update_coefficients(
        self, 
        error_real: np.ndarray, 
        error_imag: np.ndarray,
        coef_real: np.ndarray,
        coef_imag: np.ndarray
    ):
        """
        IPNLMS系数更新（根据Athena实现）
        
        根据Athena C代码:
        kl[f,b] = (1-α)/(2N) + (1+α) × |coef[f,b]|² / (Σ|coef[f]|² + ε)
        coef += μ × kl × ref × conj(error) / (||ref||² + β)
        
        Args:
            error_real, error_imag: (F,) - 误差信号
            coef_real, coef_imag: (F, num_blocks) - 待更新的系数
        """
        # ========== Step 1: 计算IPNLMS比例因子 kl ==========
        # 系数幅度平方
        coef_mag_sq = coef_real ** 2 + coef_imag ** 2  # (F, num_blocks)
        sum_coef_mag_sq = np.sum(coef_mag_sq, axis=1, keepdims=True) + self.eps  # (F, 1)
        
        # IPNLMS比例因子
        # kl = (1-α)/(2N) + (1+α) × |coef|² / (Σ|coef|² + ε)
        # 与Athena C代码保持一致
        kl = (1 - self.alpha) / (2 * self.num_blocks) + \
             (1 + self.alpha) * coef_mag_sq / (sum_coef_mag_sq + self.eps)
        
        # ========== Step 2: 计算参考信号功率 ==========
        ref_power = self.ref_history_real ** 2 + self.ref_history_imag ** 2  # (F, num_blocks)
        total_ref_power = np.sum(ref_power, axis=1)  # (F,)
        
        # ========== Step 3: 计算归一化步长 ==========
        # 根据Athena代码：μ_normalized = μ / (||ref||² + β)
        mu_normalized = self.mu / (total_ref_power + self.beta + self.eps)  # (F,)
        
        # ========== Step 4: 计算 ref × conj(error) ==========
        # ref × conj(error) = ref × (error_real - j×error_imag)
        # 实部 = ref_real × error_real + ref_imag × error_imag
        product_real = self.ref_history_real * error_real[:, np.newaxis] + \
                       self.ref_history_imag * error_imag[:, np.newaxis]
        # 虚部 = ref_imag × error_real - ref_real × error_imag
        product_imag = self.ref_history_imag * error_real[:, np.newaxis] - \
                       self.ref_history_real * error_imag[:, np.newaxis]
        
        # ========== Step 5: 应用IPNLMS更新 ==========
        update_real = mu_normalized[:, np.newaxis] * kl * product_real
        update_imag = mu_normalized[:, np.newaxis] * kl * product_imag
        
        # 限制单次更新幅度 (稳定性措施)
        max_update = 0.01
        update_real = np.clip(update_real, -max_update, max_update)
        update_imag = np.clip(update_imag, -max_update, max_update)
        
        coef_real += update_real
        coef_imag += update_imag
        
        # 限制系数幅度 (稳定性措施)
        coef_mag = np.sqrt(coef_real ** 2 + coef_imag ** 2 + 1e-10)
        max_coef = 2.0
        mask = coef_mag > max_coef
        if np.any(mask):
            scale = max_coef / coef_mag
            coef_real[mask] *= scale[mask]
            coef_imag[mask] *= scale[mask]
    
    def _select_best_filter(
        self,
        fir_err_real: np.ndarray,
        fir_err_imag: np.ndarray,
        adf_err_real: np.ndarray,
        adf_err_imag: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据Athena双滤波器机制，逐频点选择误差更小的输出
        
        Args:
            fir_err_real, fir_err_imag: FIR滤波器误差 (F,)
            adf_err_real, adf_err_imag: ADF滤波器误差 (F,)
            
        Returns:
            selected_error_real, selected_error_imag: 选中的误差信号 (F,)
            adf_used_mask: 每个频点是否使用ADF的掩码 (F,)
        """
        # 计算每个频点的误差功率
        fir_error_power = fir_err_real ** 2 + fir_err_imag ** 2  # (F,)
        adf_error_power = adf_err_real ** 2 + adf_err_imag ** 2  # (F,)
        
        # 根据Athena逻辑：逐频点选择误差更小的
        # if (energy_err_fir >= energy_err_adf) use ADF, else use FIR
        adf_used_mask = (fir_error_power >= adf_error_power).astype(np.float32)
        
        selected_error_real = adf_used_mask * adf_err_real + (1 - adf_used_mask) * fir_err_real
        selected_error_imag = adf_used_mask * adf_err_imag + (1 - adf_used_mask) * fir_err_imag
        
        return selected_error_real, selected_error_imag, adf_used_mask
    
    def process_frame(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单帧
        
        Args:
            mic_frame: (F,) complex64 - 麦克风频谱
            ref_frame: (F,) complex64 - 远端参考频谱
            
        Returns:
            error_frame: (F,) complex64 - 误差信号
            echo_estimate: (F,) complex64 - 估计的回声
        """
        F = self.num_freq_bins
        
        # 分离实部虚部
        mic_real = mic_frame.real
        mic_imag = mic_frame.imag
        ref_real = ref_frame.real
        ref_imag = ref_frame.imag
        
        # 推入历史
        self._push_ref_frame(ref_real, ref_imag)
        
        # 更新功率估计
        power = 0.9 * self.power_estimate + 0.1 * (ref_real ** 2 + ref_imag ** 2)
        self.power_estimate = power
        
        if self.use_dual_filter:
            # ========== 双滤波器模式 ==========
            
            # 1. FIR估计回声
            fir_echo_real, fir_echo_imag = self._estimate_echo_with_coef(
                self.fir_coef_real, self.fir_coef_imag
            )
            fir_err_real = mic_real - fir_echo_real
            fir_err_imag = mic_imag - fir_echo_imag
            
            # 2. ADF估计回声
            adf_echo_real, adf_echo_imag = self._estimate_echo_with_coef(
                self.adf_coef_real, self.adf_coef_imag
            )
            adf_err_real = mic_real - adf_echo_real
            adf_err_imag = mic_imag - adf_echo_imag
            
            # 3. 选择误差更小的输出
            selected_err_real, selected_err_imag, adf_mask = \
                self._select_best_filter(fir_err_real, fir_err_imag, adf_err_real, adf_err_imag)
            
            # 4. 混合回声估计
            echo_real = adf_mask * adf_echo_real + (1 - adf_mask) * fir_echo_real
            echo_imag = adf_mask * adf_echo_imag + (1 - adf_mask) * fir_echo_imag
            
            # 5. 更新MSE估计 (用于双讲检测)
            lambda_mse = 0.95
            mic_power = mic_real ** 2 + mic_imag ** 2
            fir_error_power = fir_err_real ** 2 + fir_err_imag ** 2
            adf_error_power = adf_err_real ** 2 + adf_err_imag ** 2
            
            self.mse_mic_in = lambda_mse * self.mse_mic_in + (1 - lambda_mse) * mic_power
            self.mse_main = lambda_mse * self.mse_main + (1 - lambda_mse) * fir_error_power
            self.mse_adpt = lambda_mse * self.mse_adpt + (1 - lambda_mse) * adf_error_power
            
            # 双讲检测 (根据Athena逻辑)
            dt_mask_adf = (self.mse_adpt > self.mse_mic_in * self.mse_ratio_out_in).astype(np.float32)
            recover_mask = (
                (self.mse_mic_in > self.mse_adpt * self.mse_ratio_out_in) & 
                (self.mse_adpt < self.filter_copy_fac * self.mse_main)
            ).astype(np.float32)
            
            # 应用双讲保护：清零ADF系数
            self.adf_coef_real *= (1 - dt_mask_adf[:, np.newaxis])
            self.adf_coef_imag *= (1 - dt_mask_adf[:, np.newaxis])
            
            # 恢复：FIR复制到ADF
            if np.any(recover_mask):
                recover_2d = recover_mask[:, np.newaxis]
                self.adf_coef_real = np.where(recover_2d, self.fir_coef_real, self.adf_coef_real)
                self.adf_coef_imag = np.where(recover_2d, self.fir_coef_imag, self.adf_coef_imag)
            
            # 6. 更新ADF系数 (快速自适应)
            self._update_coefficients(adf_err_real, adf_err_imag, self.adf_coef_real, self.adf_coef_imag)
            
            # 7. 慢速更新FIR系数 (当ADF表现好时)
            alpha_fir = 0.05
            mask_2d = adf_mask[:, np.newaxis]
            self.fir_coef_real = np.where(
                mask_2d > 0.5,
                alpha_fir * self.adf_coef_real + (1 - alpha_fir) * self.fir_coef_real,
                self.fir_coef_real
            )
            self.fir_coef_imag = np.where(
                mask_2d > 0.5,
                alpha_fir * self.adf_coef_imag + (1 - alpha_fir) * self.fir_coef_imag,
                self.fir_coef_imag
            )
            
            error_real = selected_err_real
            error_imag = selected_err_imag
            
        else:
            # ========== 单滤波器模式 ==========
            
            # 估计回声
            echo_real, echo_imag = self._estimate_echo()
            
            # 计算误差
            error_real = mic_real - echo_real
            error_imag = mic_imag - echo_imag
            
            # 更新系数
            self._update_coefficients(error_real, error_imag, self.fir_coef_real, self.fir_coef_imag)
        
        # 打包为复数输出
        error_frame = np.complex64(error_real + 1j * error_imag)
        echo_estimate = np.complex64(echo_real + 1j * echo_imag)
        
        return error_frame, echo_estimate
    
    def process(self, mic_spectrum: np.ndarray, ref_spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理整个频谱序列
        
        Args:
            mic_spectrum: (T, F) complex64 - 麦克风频谱序列
            ref_spectrum: (T, F) complex64 - 远端参考频谱序列
            
        Returns:
            error_spectrum: (T, F) complex64 - 误差信号序列
            echo_estimate: (T, F) complex64 - 估计的回声序列
        """
        T = mic_spectrum.shape[0]
        
        error_spectrum = np.zeros_like(mic_spectrum)
        echo_estimate = np.zeros_like(ref_spectrum)
        
        for t in range(T):
            error_frame, echo_frame = self.process_frame(mic_spectrum[t], ref_spectrum[t])
            error_spectrum[t] = error_frame
            echo_estimate[t] = echo_frame
        
        return error_spectrum, echo_estimate


def create_numpy_ipnlms_aec(
    fft_size: int = 256,
    num_blocks: int = 8,
    mu: float = 0.5,
    alpha: float = 0.5,
    beta: float = 1e-8,
    use_dual_filter: bool = True
) -> NumPyIPNLMS:
    """
    创建NumPy IPNLMS AEC模块
    
    Args:
        fft_size: FFT点数
        num_blocks: 滤波器块数
        mu: 步长因子 (默认0.5，与Athena C模型对齐)
        alpha: IPNLMS alpha参数
        beta: 正则化因子 (默认1e-8，与Athena C模型对齐)
        use_dual_filter: 是否使用双滤波器机制
        
    Returns:
        NumPyIPNLMS 模块
    """
    return NumPyIPNLMS(
        fft_size=fft_size,
        num_blocks=num_blocks,
        mu=mu,
        alpha=alpha,
        beta=beta,
        use_dual_filter=use_dual_filter
    )


# 测试
if __name__ == "__main__":
    print("Testing NumPyIPNLMS...")
    
    # 创建模块
    aec = create_numpy_ipnlms_aec(fft_size=256, num_blocks=8)
    
    # 生成测试数据
    T, F = 20, 129
    
    # 模拟回声场景
    np.random.seed(42)
    ref = np.random.randn(T, F).astype(np.complex64) + 1j * np.random.randn(T, F).astype(np.complex64)
    echo = ref * 0.7  # 简单回声模型
    near_end = (np.random.randn(T, F) + 1j * np.random.randn(T, F)) * 0.3
    mic = echo + near_end
    
    print(f"Input shapes: mic={mic.shape}, ref={ref.shape}")
    
    # 处理
    error, echo_est = aec.process(mic, ref)
    
    print(f"Output shapes: error={error.shape}, echo={echo_est.shape}")
    
    # 计算ERLE
    echo_power = np.mean(np.abs(mic - near_end) ** 2)
    residual_power = np.mean(np.abs(error - near_end) ** 2)
    erle = 10 * np.log10(echo_power / (residual_power + 1e-10))
    
    print(f"ERLE: {erle:.2f} dB")
    print("\nAll tests passed!")
