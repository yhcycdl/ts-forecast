# losses/registry.py
import torch
import torch.nn as nn


def _align_pred_target(pred: torch.Tensor, target: torch.Tensor):
    """
    对齐 pred/target 的常见形状差异，避免因为维度不一致报错。
    支持：
      pred:   (B,1,P) 或 (B,P) 或 (B,P,1) 或 (B,P,C)
      target: (B,P,1) 或 (B,1,P) 或 (B,P) 或 (B,P,C)
    返回：shape 尽量与 pred 一致的 (pred, target)
    """
    # pred/target 都是三维且刚好互为转置: (B,C,P) <-> (B,P,C)
    if pred.dim() == 3 and target.dim() == 3:
        if pred.shape[1] == target.shape[2] and pred.shape[2] == target.shape[1]:
            target = target.transpose(1, 2).contiguous()

    # pred: (B,1,P)
    if pred.dim() == 3 and pred.shape[1] == 1:
        if target.dim() == 3 and target.shape[2] == 1:          # (B,P,1) -> (B,1,P)
            target = target.transpose(1, 2).contiguous()
        elif target.dim() == 2:                                  # (B,P) -> (B,1,P)
            target = target.unsqueeze(1)

    # pred: (B,P)（二维）
    elif pred.dim() == 2:
        if target.dim() == 3 and target.shape[-1] == 1:          # (B,P,1) -> (B,P)
            target = target.squeeze(-1)
        elif target.dim() == 3 and target.shape[1] == 1:         # (B,1,P) -> (B,P)
            target = target.squeeze(1)

    return pred, target


def _infer_time_dim(pred: torch.Tensor, x_input: torch.Tensor | None = None) -> int:
    if pred.dim() <= 1:
        raise ValueError(f"Prediction tensor must have at least 2 dims, got {tuple(pred.shape)}")
    if pred.dim() == 2:
        return 1
    if pred.dim() != 3:
        raise ValueError(f"Unsupported prediction shape for HybridLoss: {tuple(pred.shape)}")

    if x_input is not None and x_input.dim() == 3:
        in_channels = int(x_input.shape[1])
        if pred.shape[1] == in_channels and pred.shape[2] != in_channels:
            return 2  # (B,C,P)
        return 1      # 默认按 (B,P,C)

    if pred.shape[1] == 1 and pred.shape[2] > 1:
        return 2
    return 1


def _first_step(pred: torch.Tensor, time_dim: int) -> torch.Tensor:
    if pred.dim() == 2:
        return pred[:, :1]
    if time_dim == 1:
        return pred[:, 0, :]
    return pred[:, :, 0]


class HybridLoss(nn.Module):
    """
    混合 loss：时间域 L1 + 导数 MSE + 频域幅值/相位 L1 + 边界斜率连续性
    注意：需要额外输入 x_input（历史窗口）来做连续性约束
    """
    def __init__(self, fft_weight=0.1, deriv_weight=1.0, cont_weight=5.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.fft_weight = float(fft_weight)
        self.deriv_weight = float(deriv_weight)
        self.cont_weight = float(cont_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, x_input: torch.Tensor):
        pred, target = _align_pred_target(pred, target)
        time_dim = _infer_time_dim(pred, x_input)

        # 1) 时间域
        loss_time = self.l1(pred, target)

        # 2) 导数（按真实时间维求差分）
        pred_d1 = torch.diff(pred, dim=time_dim)
        target_d1 = torch.diff(target, dim=time_dim)
        loss_deriv = self.mse(pred_d1, target_d1)

        # 3) 频域（幅值 + 相位）
        pred_fft = torch.fft.rfft(pred, dim=time_dim)
        target_fft = torch.fft.rfft(target, dim=time_dim)
        loss_freq = self.l1(torch.abs(pred_fft), torch.abs(target_fft)) + \
                    self.l1(torch.angle(pred_fft), torch.angle(target_fft))

        # 4) 边界斜率连续性：hist 最后斜率 vs pred 第一步斜率
        hist_last = x_input[..., -1]
        hist_prev = x_input[..., -2]
        pred_first = _first_step(pred, time_dim)

        if hist_last.dim() == 1:
            hist_last = hist_last.unsqueeze(-1)
            hist_prev = hist_prev.unsqueeze(-1)
        if pred_first.dim() == 1:
            pred_first = pred_first.unsqueeze(-1)

        # 当输出通道数少于输入通道数时，默认取前几个输入通道对齐。
        # 当前数据集列顺序是 P1..P7,Q，预测 P1 时这会正确落在第 0 通道。
        out_channels = int(pred_first.shape[1])
        hist_last = hist_last[:, :out_channels]
        hist_prev = hist_prev[:, :out_channels]

        hist_slope = hist_last - hist_prev
        pred_slope = pred_first - hist_last
        loss_cont = self.mse(pred_slope, hist_slope)

        return loss_time + self.deriv_weight * loss_deriv + self.fft_weight * loss_freq + self.cont_weight * loss_cont


class WeightedMSE(nn.Module):
    """
    weight = 1 + alpha * |y|
    用于强调尖峰（target 越大，惩罚越大）
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = _align_pred_target(pred, target)
        w = 1.0 + self.alpha * torch.abs(target)
        return torch.mean(w * (pred - target) ** 2)


def build_criterion(args):
    name = args.loss.lower()

    if name == "hybrid":
        return HybridLoss(
            fft_weight=getattr(args, "fft_weight", 0.1),
            deriv_weight=getattr(args, "deriv_weight", 1.0),
            cont_weight=getattr(args, "cont_weight", 5.0),
        )

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    if name == "huber":
        return nn.SmoothL1Loss(beta=float(getattr(args, "huber_beta", 0.3)))

    if name == "wmse":
        return WeightedMSE(alpha=float(getattr(args, "wmse_alpha", 1.0)))

    raise ValueError(f"Unknown loss: {args.loss}")
