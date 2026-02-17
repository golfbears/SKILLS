"""
PFB (Polyphase Filter Bank) 时频变换库

基于验证正确的 C 模型实现，提供 NumPy 和 PyTorch 版本。
参数：FFT_LEN=256, WIN_LEN=768, FRM_LEN=128, Ppf_tap=3, Scale=-256.0
群延时补偿：-648 采样 @ 16kHz
"""

from .pfb_analysis import PFBAnalysis
from .pfb_synthesis import PFBSynthesis
from .pfb_pytorch import PFBAnalysisLayer, PFBSynthesisLayer, PFBTransform

__version__ = "2.0.0"
__all__ = [
    "PFBAnalysis",
    "PFBSynthesis",
    "PFBAnalysisLayer",
    "PFBSynthesisLayer",
    "PFBTransform"
]
