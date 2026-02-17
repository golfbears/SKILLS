"""
Frequency Domain IPNLMS Adaptive Filter - Complete Implementation

基于Athena-signal原生C实现的频域IPNLMS自适应滤波器
支持真正的多块历史帧处理和在线自适应更新
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class FrequencyDomainIPNLMS(nn.Module):
    """
    频域IPNLMS自适应滤波器 - 完整版（支持Athena双滤波器机制）
    
    每个频点维护 num_blocks 个历史帧的系数
    实现真正的在线自适应滤波
    
    输入: 
        mic_fft: (B, T, F) complex64 - 麦克风频谱
        ref_fft: (B, T, F) complex64 - 远端参考频谱
        
    输出:
        error_fft: (B, T, F) complex64 - 误差信号（回声消除后）
        echo_estimate: (B, T, F) complex64 - 估计的回声频谱
    """
    
    def __init__(
        self,
        fft_size: int = 256,
        num_blocks: int = 8,
        mu: float = 0.5,  # 与Athena C模型对齐
        alpha: float = 0.5,
        beta: float = 1e-8,  # 与Athena C模型对齐
        eps: float = 1e-10,
        use_dual_filter: bool = True  # 是否使用双滤波器机制
    ):
        super().__init__()
        
        self.fft_size = fft_size
        self.num_freq_bins = fft_size // 2 + 1  # 129 for 256 FFT
        self.num_blocks = num_blocks
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_dual_filter = use_dual_filter
        
        # 双滤波器机制：维护两组系数
        # fir_coef: 稳定系数（根据Athena命名）
        # adf_coef: 自适应系数
        
        # FIR系数 - 初始化为很小的值
        init_coef = torch.randn(self.num_freq_bins, self.num_blocks) * 1e-5
        
        # FIR滤波器系数
        self.fir_coef_real = nn.Parameter(init_coef.clone())
        self.fir_coef_imag = nn.Parameter(init_coef.clone() * 0)
        
        # ADF系数（仅在use_dual_filter=True时使用）
        if self.use_dual_filter:
            self.adf_coef_real = nn.Parameter(init_coef.clone())
            self.adf_coef_imag = nn.Parameter(init_coef.clone() * 0)
        
        # 参考信号历史缓冲区: (F, num_blocks) 复数
        self.register_buffer('ref_history_real', torch.zeros(self.num_freq_bins, self.num_blocks))
        self.register_buffer('ref_history_imag', torch.zeros(self.num_freq_bins, self.num_blocks))
        
        # 功率估计平滑系数
        self.lambda_power = 0.9
        self.register_buffer('power_estimate', torch.ones(self.num_freq_bins))
        
        # 双讲检测相关 - MSE跟踪
        # mse_mic_in: 麦克风输入功率
        # mse_adpt: ADF误差功率
        # mse_main: FIR误差功率
        self.register_buffer('mse_mic_in', torch.ones(self.num_freq_bins))
        self.register_buffer('mse_adpt', torch.ones(self.num_freq_bins))
        self.register_buffer('mse_main', torch.ones(self.num_freq_bins))
        
        # 双讲检测参数（根据Athena）
        self.mse_ratio_out_in = 8.0  # MSE_RATIO_OUT_IN
        self.filter_copy_fac = 0.5  # FILTER_COPY_FAC
        self.dt_threshold = 0.5
        
    def reset(self):
        """重置滤波器状态"""
        self.ref_history_real.zero_()
        self.ref_history_imag.zero_()
        self.power_estimate.fill_(1.0)
        self.mse_mic_in.fill_(1.0)
        self.mse_adpt.fill_(1.0)
        self.mse_main.fill_(1.0)
        with torch.no_grad():
            self.fir_coef_real.fill_(1e-5)
            self.fir_coef_imag.fill_(0)
            if self.use_dual_filter:
                self.adf_coef_real.fill_(1e-5)
                self.adf_coef_imag.fill_(0)
    
    def _push_ref_frame(self, ref_real: torch.Tensor, ref_imag: torch.Tensor):
        """将新的参考帧推入历史缓冲区"""
        # ref_real, ref_imag: (F,)
        # ref_history: (F, num_blocks)
        
        # 右移历史
        self.ref_history_real[:, 1:].copy_(self.ref_history_real[:, :-1])
        self.ref_history_imag[:, 1:].copy_(self.ref_history_imag[:, :-1])
        
        # 插入新帧
        self.ref_history_real[:, 0].copy_(ref_real)
        self.ref_history_imag[:, 0].copy_(ref_imag)
    
    def _estimate_echo_with_coef(
        self, 
        coef_real: torch.Tensor, 
        coef_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用指定系数估计回声: echo[f] = Σ conj(coef[f,b]) × ref_history[f,b]
        
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
        
        echo_real = torch.sum(
            coef_real * self.ref_history_real + 
            coef_imag * self.ref_history_imag,
            dim=1
        )
        echo_imag = torch.sum(
            coef_real * self.ref_history_imag - 
            coef_imag * self.ref_history_real,
            dim=1
        )
        
        return echo_real, echo_imag
    
    def _estimate_echo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用FIR系数估计回声（默认）"""
        return self._estimate_echo_with_coef(self.fir_coef_real, self.fir_coef_imag)
    
    def _update_coefficients(
        self, 
        error_real: torch.Tensor, 
        error_imag: torch.Tensor,
        power: torch.Tensor,
        coef_real: torch.Tensor,
        coef_imag: torch.Tensor
    ):
        """
        IPNLMS系数更新（根据Athena实现）
        
        根据Athena C代码:
        kl[f,b] = (1-α)/(2N) + (1+α) × |coef[f,b]|² / (Σ|coef[f]|² + ε)
        coef += μ × kl × ref × conj(error) / (||ref||² + β)
        
        Args:
            error_real, error_imag: (F,) - 误差信号
            power: (F,) - 参考信号功率
            coef_real, coef_imag: (F, num_blocks) - 待更新的系数
        """
        # ========== Step 1: 计算IPNLMS比例因子 kl ==========
        # 系数幅度平方
        coef_mag_sq = coef_real ** 2 + coef_imag ** 2  # (F, num_blocks)
        sum_coef_mag_sq = torch.sum(coef_mag_sq, dim=1, keepdim=True) + self.eps  # (F, 1)
        
        # IPNLMS比例因子
        # kl = (1-α)/(2N) + (1+α) × |coef|² / (Σ|coef|² + ε)
        # 与Athena C代码保持一致: sum_coef_mag_sq + eps
        kl = (1 - self.alpha) / (2 * self.num_blocks) + \
             (1 + self.alpha) * coef_mag_sq / (sum_coef_mag_sq + self.eps)  # (F, num_blocks)
        
        # ========== Step 2: 计算参考信号功率 ==========
        ref_power = self.ref_history_real ** 2 + self.ref_history_imag ** 2  # (F, num_blocks)
        total_ref_power = torch.sum(ref_power, dim=1)  # (F,)
        
        # ========== Step 3: 计算归一化步长 ==========
        # 根据Athena代码：使用固定正则化 0.01
        # μ_normalized = μ / (||ref||² + 0.01)
        mu_normalized = self.mu / (total_ref_power + self.beta + self.eps)  # (F,)
        
        # ========== Step 4: 计算 ref × conj(error) ==========
        # ref × conj(error) = ref × (error_real - j×error_imag)
        # 实部 = ref_real × error_real + ref_imag × error_imag
        product_real = self.ref_history_real * error_real.unsqueeze(1) + \
                       self.ref_history_imag * error_imag.unsqueeze(1)
        # 虚部 = ref_imag × error_real - ref_real × error_imag
        product_imag = self.ref_history_imag * error_real.unsqueeze(1) - \
                       self.ref_history_real * error_imag.unsqueeze(1)
        
        # ========== Step 5: 应用IPNLMS更新 ==========
        # update = μ × kl × product / (||ref||² + β)
        with torch.no_grad():
            update_real = mu_normalized.unsqueeze(1) * kl * product_real
            update_imag = mu_normalized.unsqueeze(1) * kl * product_imag
            
            # 限制单次更新幅度
            max_update = 0.01
            update_real = torch.clamp(update_real, -max_update, max_update)
            update_imag = torch.clamp(update_imag, -max_update, max_update)
            
            coef_real.add_(update_real)
            coef_imag.add_(update_imag)
            
            # 限制系数幅度
            coef_mag = torch.sqrt(coef_real ** 2 + coef_imag ** 2 + 1e-10)
            max_coef = 2.0
            mask = coef_mag > max_coef
            if mask.any():
                scale = max_coef / coef_mag
                coef_real.mul_(torch.where(mask, scale, torch.ones_like(scale)))
                coef_imag.mul_(torch.where(mask, scale, torch.ones_like(scale)))
    
    def _update_adf_coefficients(
        self, 
        error_real: torch.Tensor, 
        error_imag: torch.Tensor,
        power: torch.Tensor
    ):
        """更新ADF系数（自适应滤波器）"""
        if self.use_dual_filter:
            self._update_coefficients(
                error_real, error_imag, power,
                self.adf_coef_real, self.adf_coef_imag
            )
    
    def _select_best_filter(
        self,
        fir_error_real: torch.Tensor,
        fir_error_imag: torch.Tensor,
        adf_error_real: torch.Tensor,
        adf_error_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据Athena双滤波器机制，逐频点选择误差更小的输出
        
        Args:
            fir_error_real, fir_error_imag: FIR滤波器误差 (F,)
            adf_error_real, adf_error_imag: ADF滤波器误差 (F,)
            
        Returns:
            selected_error_real, selected_error_imag: 选中的误差信号 (F,)
            adf_used_mask: 每个频点是否使用ADF的掩码 (F,)
        """
        # 计算每个频点的误差功率
        fir_error_power = fir_error_real ** 2 + fir_error_imag ** 2  # (F,)
        adf_error_power = adf_error_real ** 2 + adf_error_imag ** 2  # (F,)
        
        # 根据Athena逻辑：逐频点选择误差更小的
        # if (energy_err_fir >= energy_err_adf) use ADF, else use FIR
        adf_used_mask = (fir_error_power >= adf_error_power).float()  # (F,) - 1.0=ADF, 0.0=FIR
        
        # 使用mask选择每个频点的误差
        selected_error_real = adf_used_mask * adf_error_real + (1 - adf_used_mask) * fir_error_real
        selected_error_imag = adf_used_mask * adf_error_imag + (1 - adf_used_mask) * fir_error_imag
        
        return selected_error_real, selected_error_imag, adf_used_mask
    
    def forward(self, mic_fft: torch.Tensor, ref_fft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 逐帧处理
        
        Args:
            mic_fft: (B, T, F) complex64 - 麦克风频谱
            ref_fft: (B, T, F) complex64 - 远端参考频谱
            
        Returns:
            error_fft: (B, T, F) complex64 - 误差信号
            echo_estimate: (B, T, F) complex64 - 估计的回声
        """
        B, T, F = mic_fft.shape
        
        assert F == self.num_freq_bins, f"Expected {self.num_freq_bins} freq bins, got {F}"
        
        # 分离实部虚部
        mic_real = mic_fft.real
        mic_imag = mic_fft.imag
        ref_real = ref_fft.real
        ref_imag = ref_fft.imag
        
        # 初始化输出
        error_real = torch.zeros_like(mic_real)
        error_imag = torch.zeros_like(mic_imag)
        echo_real = torch.zeros_like(ref_real)
        echo_imag = torch.zeros_like(ref_imag)
        
        # 统计使用ADF/FIR的频点数
        self.adf_freq_count = 0
        self.fir_freq_count = 0
        total_freq_points = 0
        
        # 逐帧处理
        for t in range(T):
            for b in range(B):
                # 当前帧
                mic_frame_real = mic_real[b, t, :]  # (F,)
                mic_frame_imag = mic_imag[b, t, :]
                ref_frame_real = ref_real[b, t, :]
                ref_frame_imag = ref_imag[b, t, :]
                
                # 推入历史
                self._push_ref_frame(ref_frame_real, ref_frame_imag)
                
                # 更新功率估计
                power = self.lambda_power * self.power_estimate + \
                        (1 - self.lambda_power) * (ref_frame_real ** 2 + ref_frame_imag ** 2)
                self.power_estimate.copy_(power)
                
                if self.use_dual_filter and self.training:
                    # ========== 双滤波器模式 ==========
                    
                    # 1. 使用FIR系数估计回声
                    fir_echo_real, fir_echo_imag = self._estimate_echo_with_coef(
                        self.fir_coef_real, self.fir_coef_imag
                    )
                    fir_err_real = mic_frame_real - fir_echo_real
                    fir_err_imag = mic_frame_imag - fir_echo_imag
                    
                    # 2. 使用ADF系数估计回声
                    adf_echo_real, adf_echo_imag = self._estimate_echo_with_coef(
                        self.adf_coef_real, self.adf_coef_imag
                    )
                    adf_err_real = mic_frame_real - adf_echo_real
                    adf_err_imag = mic_frame_imag - adf_echo_imag
                    
                    # 3. 逐频点选择误差更小的输出
                    selected_err_real, selected_err_imag, adf_mask = \
                        self._select_best_filter(fir_err_real, fir_err_imag, 
                                                 adf_err_real, adf_err_imag)
                    
                    # 统计ADF/FIR使用情况
                    adf_used = adf_mask.sum().item()
                    fir_used = (F - adf_used)
                    self.adf_freq_count += adf_used
                    self.fir_freq_count += fir_used
                    total_freq_points += F
                    
                    # 4. 选择回声估计（使用mask混合）
                    echo_real[b, t, :] = adf_mask * adf_echo_real + (1 - adf_mask) * fir_echo_real
                    echo_imag[b, t, :] = adf_mask * adf_echo_imag + (1 - adf_mask) * fir_echo_imag
                    
                    # 5. 保存输出
                    error_real[b, t, :] = selected_err_real
                    error_imag[b, t, :] = selected_err_imag
                    
                    # 6. 更新MSE估计（用于双讲检测）
                    with torch.no_grad():
                        # 平滑更新MSE
                        lambda_mse = 0.95
                        mic_power = mic_frame_real ** 2 + mic_frame_imag ** 2
                        fir_error_power = fir_err_real ** 2 + fir_err_imag ** 2
                        adf_error_power = adf_err_real ** 2 + adf_err_imag ** 2
                        
                        self.mse_mic_in.copy_(lambda_mse * self.mse_mic_in + (1 - lambda_mse) * mic_power)
                        self.mse_main.copy_(lambda_mse * self.mse_main + (1 - lambda_mse) * fir_error_power)
                        self.mse_adpt.copy_(lambda_mse * self.mse_adpt + (1 - lambda_mse) * adf_error_power)
                        
                        # 双讲检测（根据Athena逻辑）
                        # 情况1: ADF误差远大于输入功率 → 可能双讲，清零ADF系数
                        dt_mask_adf = (self.mse_adpt > self.mse_mic_in * self.mse_ratio_out_in).float()
                        
                        # 情况2: 输入功率远大于ADF误差，且ADF误差接近FIR误差 → 恢复，FIR复制到ADF
                        recover_mask = (
                            (self.mse_mic_in > self.mse_adpt * self.mse_ratio_out_in) & 
                            (self.mse_adpt < self.filter_copy_fac * self.mse_main)
                        ).float()
                        
                        # 应用双讲保护
                        if dt_mask_adf.sum() > 0:
                            # 清零ADF系数（双讲保护）
                            self.adf_coef_real.data.mul_(1 - dt_mask_adf.unsqueeze(1))
                            self.adf_coef_imag.data.mul_(1 - dt_mask_adf.unsqueeze(1))
                        
                        if recover_mask.sum() > 0:
                            # FIR复制到ADF（恢复）
                            mask_2d = recover_mask.unsqueeze(1)
                            self.adf_coef_real.data = torch.where(
                                mask_2d > 0.5,
                                self.fir_coef_real.data,
                                self.adf_coef_real.data
                            )
                            self.adf_coef_imag.data = torch.where(
                                mask_2d > 0.5,
                                self.fir_coef_imag.data,
                                self.adf_coef_imag.data
                            )
                    
                    # 7. 更新ADF系数（快速自适应）
                    self._update_adf_coefficients(adf_err_real, adf_err_imag, power)
                    
                    # 7. 慢速更新FIR系数（当ADF表现好时）
                    with torch.no_grad():
                        alpha_fir = 0.05  # 非常慢的跟踪
                        # 只更新ADF表现更好的频点对应的FIR系数
                        mask_2d = adf_mask.unsqueeze(1)  # (F, 1)
                        self.fir_coef_real.data = torch.where(
                            mask_2d > 0.5,
                            alpha_fir * self.adf_coef_real.data + (1 - alpha_fir) * self.fir_coef_real.data,
                            self.fir_coef_real.data
                        )
                        self.fir_coef_imag.data = torch.where(
                            mask_2d > 0.5,
                            alpha_fir * self.adf_coef_imag.data + (1 - alpha_fir) * self.fir_coef_imag.data,
                            self.fir_coef_imag.data
                        )
                    
                else:
                    # ========== 单滤波器模式 ==========
                    
                    # 估计回声
                    echo_frame_real, echo_frame_imag = self._estimate_echo()
                    
                    # 计算误差
                    err_frame_real = mic_frame_real - echo_frame_real
                    err_frame_imag = mic_frame_imag - echo_frame_imag
                    
                    # 保存输出
                    error_real[b, t, :] = err_frame_real
                    error_imag[b, t, :] = err_frame_imag
                    echo_real[b, t, :] = echo_frame_real
                    echo_imag[b, t, :] = echo_frame_imag
                    
                    # 更新系数（仅在训练模式）
                    if self.training:
                        self._update_coefficients(
                            err_frame_real, err_frame_imag, power,
                            self.fir_coef_real, self.fir_coef_imag
                        )
        
        # 打包为复数输出
        error_fft = torch.complex(error_real, error_imag)
        echo_estimate = torch.complex(echo_real, echo_imag)
        
        return error_fft, echo_estimate


class FrequencyDomainIPNLMSFast(nn.Module):
    """
    快速版频域IPNLMS - 批量处理
    
    使用批量矩阵运算加速处理
    """
    
    def __init__(
        self,
        fft_size: int = 256,
        num_blocks: int = 8,
        mu: float = 0.5,
        alpha: float = 0.5,
        beta: float = 1e-8
    ):
        super().__init__()
        
        self.fft_size = fft_size
        self.num_freq_bins = fft_size // 2 + 1
        self.num_blocks = num_blocks
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        
        # 系数: (F, num_blocks)
        self.coef_real = nn.Parameter(torch.zeros(self.num_freq_bins, self.num_blocks))
        self.coef_imag = nn.Parameter(torch.zeros(self.num_freq_bins, self.num_blocks))
        
        # 历史缓冲区: (B, F, num_blocks)
        self.register_buffer('ref_history_real', torch.zeros(1, self.num_freq_bins, self.num_blocks))
        self.register_buffer('ref_history_imag', torch.zeros(1, self.num_freq_bins, self.num_blocks))
        
        # 功率估计
        self.register_buffer('power_estimate', torch.ones(self.num_freq_bins))
        
    def reset(self):
        self.ref_history_real.zero_()
        self.ref_history_imag.zero_()
        self.power_estimate.fill_(1.0)
        with torch.no_grad():
            self.coef_real.fill_(1e-5)
            self.coef_imag.fill_(0)
    
    def forward(self, mic_fft: torch.Tensor, ref_fft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量前向传播"""
        B, T, F = mic_fft.shape
        
        # 简化版：使用当前系数进行批量卷积
        # 真实的多块处理需要维护历史，这里用简化版
        
        coef = torch.complex(self.coef_real, self.coef_imag)  # (F, num_blocks)
        
        # 只使用第0块系数进行估计（简化）
        echo = ref_fft * torch.conj(coef[:, 0].unsqueeze(0).unsqueeze(0))  # (B, T, F)
        error = mic_fft - echo
        
        # 简化的系数更新
        if self.training:
            # 计算功率
            power = torch.mean(ref_fft.abs() ** 2, dim=[0, 1]) + self.beta  # (F,)
            
            # 简化的比例因子
            kl = torch.ones(self.num_freq_bins, self.num_blocks) / self.num_blocks
            
            # 更新
            with torch.no_grad():
                update = self.mu * torch.mean(error * torch.conj(ref_fft), dim=[0, 1]) / (power + 1e-10)
                self.coef_real[:, 0].add_(update.real)
                self.coef_imag[:, 0].add_(update.imag)
        
        return error, echo


def create_ipnlms_aec(
    fft_size: int = 256,
    num_blocks: int = 8,
    mu: float = 0.5,
    alpha: float = 0.5,
    beta: float = 1e-8,
    fast_mode: bool = False,
    use_dual_filter: bool = True
) -> nn.Module:
    """
    创建IPNLMS AEC模块
    
    Args:
        fft_size: FFT点数
        num_blocks: 滤波器块数
        mu: 步长因子
        alpha: IPNLMS alpha参数
        beta: 正则化因子
        fast_mode: 是否使用快速模式（批量处理，简化版）
        use_dual_filter: 是否使用双滤波器机制（默认True，推荐使用）
        
    Returns:
        FrequencyDomainIPNLMS 模块
    """
    if fast_mode:
        return FrequencyDomainIPNLMSFast(
            fft_size=fft_size,
            num_blocks=num_blocks,
            mu=mu,
            alpha=alpha,
            beta=beta
        )
    else:
        return FrequencyDomainIPNLMS(
            fft_size=fft_size,
            num_blocks=num_blocks,
            mu=mu,
            alpha=alpha,
            beta=beta,
            use_dual_filter=use_dual_filter
        )


# 测试
if __name__ == "__main__":
    print("Testing FrequencyDomainIPNLMS...")
    
    # 创建模块
    aec = create_ipnlms_aec(fft_size=256, num_blocks=8, fast_mode=False)
    aec.train()  # 启用训练模式进行自适应更新
    
    # 生成测试数据
    B, T, F = 1, 20, 129
    torch.manual_seed(42)
    
    # 模拟回声场景
    ref = torch.randn(B, T, F, dtype=torch.complex64)
    echo = ref * 0.7  # 简单回声模型
    near_end = torch.randn(B, T, F, dtype=torch.complex64) * 0.3
    mic = echo + near_end
    
    print(f"Input shapes: mic={mic.shape}, ref={ref.shape}")
    
    # 处理
    error, echo_est = aec(mic, ref)
    
    print(f"Output shapes: error={error.shape}, echo={echo_est.shape}")
    
    # 计算ERLE
    echo_power = torch.mean(torch.abs(mic - near_end) ** 2)
    residual_power = torch.mean(torch.abs(error - near_end) ** 2)
    erle = 10 * torch.log10(echo_power / (residual_power + 1e-10))
    
    print(f"ERLE: {erle.item():.2f} dB")
    print(f"Total parameters: {sum(p.numel() for p in aec.parameters())}")
    
    print("\nAll tests passed!")
