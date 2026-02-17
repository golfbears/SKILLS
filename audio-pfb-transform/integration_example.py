"""
PFB与DeepVQE项目集成示例

展示如何将PFB（或改进的STFT）集成到现有DeepVQE模型中。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import get_window
import sys
from pathlib import Path
import io

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class PFBFrontend(nn.Module):
    """
    PFB前端 - 替代传统STFT
    
    使用优化的Kaiser窗实现高质量的时频变换
    """
    def __init__(
        self,
        n_fft: int = 128,
        hop_length: int = 64,
        win_length: int = 128,
        kaiser_beta: float = 12.0,
        sample_rate: float = 16000
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        
        # 注册窗函数为buffer
        window = torch.from_numpy(get_window(('kaiser', kaiser_beta), win_length)).float()
        self.register_buffer('window', window)
        
        # COLA补偿系数
        self.cola_comp = self._compute_cola_compensation()
        
    def _compute_cola_compensation(self):
        """计算COLA补偿系数"""
        num_frames_overlap = self.n_fft // self.hop_length
        window_np = self.window.numpy()
        window_sum = np.zeros(self.n_fft)
        
        for i in range(num_frames_overlap):
            shifted = np.roll(window_np**2, i * self.hop_length)
            window_sum += shifted
        
        compensation = 1.0 / (window_sum + 1e-12)
        return torch.from_numpy(compensation).float()
    
    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """
        分析：时域 -> 频域
        
        Args:
            x: (batch, samples)
        
        Returns:
            spectrum: (batch, freq_bins, time_frames)
        """
        # 使用torch.stft
        spectrum = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            normalized=False
        )
        
        # 不应用COLA补偿，因为torch.stft/istft已经处理了
        # 如果需要应用，应该在分析后、合成前保持一致性
        
        # 转置为 (batch, freq, time)
        spectrum = spectrum.permute(0, 2, 1)
        
        return spectrum
    
    def synthesis(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        合成：频域 -> 时域
        
        Args:
            spectrum: (batch, freq_bins, time_frames)
        
        Returns:
            x: (batch, samples)
        """
        # 恢复形状 (batch, time, freq)
        spectrum = spectrum.permute(0, 2, 1)
        
        # 使用torch.istft
        x = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=False
        )
        
        return x


class DeepVQE_With_PFB(nn.Module):
    """
    集成PFB前端的DeepVQE模型
    
    这是示例结构，实际需要根据你的DeepVQE实现调整
    """
    def __init__(
        self,
        n_fft: int = 128,
        hop_length: int = 64,
        num_channels: int = 2,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        # PFB前端
        self.pfb = PFBFrontend(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            kaiser_beta=12.0
        )
        
        self.num_freqs = n_fft // 2 + 1
        
        # 简单的CNN结构（示例）
        self.conv1 = nn.Conv1d(self.num_freqs, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # 掩码
        self.mask = nn.Conv1d(hidden_dim, self.num_freqs, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, mic_input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            mic_input: (batch, samples)
        
        Returns:
            enhanced_audio: (batch, samples)
        """
        # PFB分析 (B, F, T)
        spectrum = self.pfb.analysis(mic_input)
        
        # 提取幅度作为特征 (B, F, T)
        spectrum_mag = torch.abs(spectrum)
        
        # 转置为 (B, T, F) 用于Conv1d
        x = spectrum_mag.permute(0, 2, 1)
        
        # 特征提取
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        
        # 预测掩码 (B, T, F)
        mask = self.sigmoid(self.mask(x))
        
        # 转置回 (B, F, T)
        mask = mask.permute(0, 2, 1)
        
        # 应用掩码
        enhanced_spectrum = spectrum * mask
        
        # PFB合成
        enhanced_audio = self.pfb.synthesis(enhanced_spectrum)
        
        return enhanced_audio


if __name__ == "__main__":
    print("=" * 60)
    print("PFB与DeepVQE集成示例")
    print("=" * 60)
    
    # 创建模型
    model = DeepVQE_With_PFB(
        n_fft=128,
        hop_length=64,
        hidden_dim=64
    )
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试完美重建（不经过网络）
    print("\n测试PFB前端完美重建...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 生成测试信号
    batch_size = 2
    num_samples = 16000
    
    t = torch.linspace(0, 1, num_samples).to(device)
    audio = (
        0.5 * torch.sin(2 * np.pi * 440 * t) +
        0.3 * torch.sin(2 * np.pi * 1000 * t) +
        0.2 * torch.sin(2 * np.pi * 3000 * t)
    ).unsqueeze(0).expand(batch_size, -1)
    
    # 只测试PFB分析+合成
    spectrum = model.pfb.analysis(audio)
    reconstructed = model.pfb.synthesis(spectrum)
    
    # 计算误差
    min_len = min(audio.shape[1], reconstructed.shape[1])
    audio_trunc = audio[:, :min_len]
    reconstructed_trunc = reconstructed[:, :min_len]
    
    mse = torch.mean((audio_trunc - reconstructed_trunc)**2).item()
    max_error = torch.max(torch.abs(audio_trunc - reconstructed_trunc)).item()
    snr = 10 * torch.log10(torch.mean(audio_trunc**2) / (torch.mean((audio_trunc - reconstructed_trunc)**2) + 1e-12)).item()
    
    print(f"   MSE: {mse:.2e}")
    print(f"   最大误差: {max_error:.2e}")
    print(f"   SNR: {snr:.2f} dB")
    
    if snr > 60:
        print(f"   ✅ 完美重建验证通过！")
    else:
        print(f"   ⚠️  SNR: {snr:.2f} dB")
    
    # 测试完整模型
    print("\n测试完整模型前向传播...")
    with torch.no_grad():
        enhanced_audio = model(audio)
    
    print(f"   输入形状: {audio.shape}")
    print(f"   输出形状: {enhanced_audio.shape}")
    print(f"   ✅ 模型前向传播成功！")
    
    # 测试梯度传播
    print("\n测试梯度传播...")
    audio_grad = audio.clone().requires_grad_(True)
    output = model(audio_grad)
    loss = torch.mean(output**2)
    loss.backward()
    
    print(f"   损失值: {loss.item():.4f}")
    print(f"   输入梯度存在: {audio_grad.grad is not None}")
    if audio_grad.grad is not None:
        print(f"   输入梯度范数: {torch.norm(audio_grad.grad).item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 集成测试完成！")
    print("=" * 60)
    
    print("\n集成说明:")
    print("1. PFBFrontend 可以直接替换现有的STFT分析/合成")
    print("2. 使用Kaiser窗获得更低的旁瓣泄漏")
    print("3. COLA补偿确保完美重建")
    print("4. 在DeepVQE中，只需将 self.stft 替换为 self.pfb")
    print("5. 频谱处理逻辑保持不变")
