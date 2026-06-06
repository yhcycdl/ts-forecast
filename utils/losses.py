# losses/registry.py
import torch
import torch.nn as nn


def _align_pred_target(pred: torch.Tensor, target: torch.Tensor, strict: bool = True):
    """
    对齐 pred/target 的常见形状差异，避免因为维度不一致报错。
    支持：
      pred:   (B,1,P) 或 (B,P) 或 (B,P,1) 或 (B,P,C)
      target: (B,P,1) 或 (B,1,P) 或 (B,P) 或 (B,P,C)
    返回：shape 尽量与 pred 一致的 (pred, target)。
    strict=True 时会拒绝 PyTorch 广播，避免多输出模型被单通道 target
    静默广播后得到虚假的 loss/metric。
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

    if strict and tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            "Prediction/target shape mismatch after alignment: "
            f"pred={tuple(pred.shape)}, target={tuple(target.shape)}. "
            "Check --enc_in/--c_out and explicit --input_cols/--output_cols."
        )

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
    混合 loss：时间域 L1 + 导数 MSE + 频域幅值约束 + 可选相位/边界连续性
    注意：需要额外输入 x_input（历史窗口）来做连续性约束
    """
    def __init__(self, fft_weight=0.1, deriv_weight=1.0, cont_weight=5.0, phase_weight=0.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.fft_weight = float(fft_weight)
        self.deriv_weight = float(deriv_weight)
        self.cont_weight = float(cont_weight)
        self.phase_weight = float(phase_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, x_input: torch.Tensor):
        pred, target = _align_pred_target(pred, target)
        time_dim = _infer_time_dim(pred, x_input)

        # 1) 时间域
        loss_time = self.l1(pred, target)

        # 2) 导数（按真实时间维求差分）
        pred_d1 = torch.diff(pred, dim=time_dim)
        target_d1 = torch.diff(target, dim=time_dim)
        loss_deriv = self.mse(pred_d1, target_d1)

        # 3) 频域。默认只约束幅值谱；直接 L1 比较 FFT angle 会被相位环绕干扰。
        pred_fft = torch.fft.rfft(pred, dim=time_dim)
        target_fft = torch.fft.rfft(target, dim=time_dim)
        loss_freq = self.l1(torch.abs(pred_fft), torch.abs(target_fft))
        if self.phase_weight > 0:
            phase_diff = torch.angle(pred_fft * torch.conj(target_fft))
            loss_freq = loss_freq + self.phase_weight * torch.mean(torch.abs(phase_diff))

        # 4) 边界斜率连续性：hist 最后斜率 vs pred 第一步斜率
        loss_cont = pred.new_tensor(0.0)
        if self.cont_weight > 0:
            if x_input is None:
                raise ValueError("HybridLoss cont_weight > 0 requires x_input.")
            if x_input.shape[-1] < 2:
                raise ValueError("HybridLoss continuity term requires at least 2 input time steps.")

            hist_last = x_input[..., -1]
            hist_prev = x_input[..., -2]
            pred_first = _first_step(pred, time_dim)

            if hist_last.dim() == 1:
                hist_last = hist_last.unsqueeze(-1)
                hist_prev = hist_prev.unsqueeze(-1)
            if pred_first.dim() == 1:
                pred_first = pred_first.unsqueeze(-1)

            # 连续性项只适合“输入第一批通道”和输出是同一物理量的场景。
            # raw->smooth、raw->event 等任务应把 cont_weight 设为 0。
            out_channels = int(pred_first.shape[1])
            if hist_last.shape[1] < out_channels:
                raise ValueError(
                    "HybridLoss continuity term needs at least as many input channels as output channels: "
                    f"input={hist_last.shape[1]}, output={out_channels}."
                )
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


def _to_bcp_for_loss(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"Expected tensor dim 2 or 3, got {tuple(x.shape)}")
    # Current forecasting models normally emit (B, P, C). Some older helpers
    # emit (B, C, P). Prefer explicit single-channel cues, then fall back to
    # the convention that the longer axis is time.
    if x.shape[1] == 1 and x.shape[2] > 1:
        return x
    if x.shape[2] == 1:
        return x.transpose(1, 2).contiguous()
    if x.shape[1] > x.shape[2]:
        return x.transpose(1, 2).contiguous()
    if x.shape[1] <= 16 and x.shape[2] > 16:
        return x
    # Ambiguous short multi-output horizon: assume model output convention
    # (B, P, C), because all active models in this repository return that.
    return x.transpose(1, 2).contiguous()


def _local_rms_torch(x_bcp: torch.Tensor, window: int) -> torch.Tensor:
    import torch.nn.functional as F

    window = max(1, int(window))
    if window <= 1:
        return torch.abs(x_bcp)
    left = (window - 1) // 2
    right = window - 1 - left
    padded = F.pad(x_bcp.square(), (left, right), mode="replicate")
    rms = F.avg_pool1d(padded, kernel_size=window, stride=1)
    return torch.sqrt(torch.clamp(rms, min=1e-12))


class QPHybridLoss(nn.Module):
    """Feature-aware differentiable loss for complex quasi-periodic signals.

    Terms:
      - robust point loss for the main waveform
      - derivative loss for phase/slope consistency
      - local-RMS envelope loss for AM signals
      - log spectrum magnitude loss for multi-frequency signals
      - target-driven event weighting for spike-like signals
    """

    def __init__(
        self,
        deriv_weight=0.5,
        envelope_weight=0.5,
        band_weight=0.05,
        event_weight=1.0,
        envelope_window=9,
        huber_beta=0.3,
    ):
        super().__init__()
        self.deriv_weight = float(deriv_weight)
        self.envelope_weight = float(envelope_weight)
        self.band_weight = float(band_weight)
        self.event_weight = float(event_weight)
        self.envelope_window = int(envelope_window)
        self.huber_beta = float(huber_beta)

    def _event_salience(self, target_bcp: torch.Tensor) -> torch.Tensor:
        centered = target_bcp - target_bcp.mean(dim=-1, keepdim=True)
        amp = torch.abs(centered)
        amp = amp / torch.clamp(amp.mean(dim=-1, keepdim=True), min=1e-6)
        deriv = torch.diff(target_bcp, dim=-1, prepend=target_bcp[..., :1])
        deriv = torch.abs(deriv) / torch.clamp(torch.abs(deriv).mean(dim=-1, keepdim=True), min=1e-6)
        return torch.clamp(0.5 * amp + 0.5 * deriv, min=0.0, max=10.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = _align_pred_target(pred, target)
        pred_bcp = _to_bcp_for_loss(pred)
        target_bcp = _to_bcp_for_loss(target)
        if tuple(pred_bcp.shape) != tuple(target_bcp.shape):
            raise ValueError(f"QPHybridLoss shape mismatch: pred={tuple(pred_bcp.shape)}, target={tuple(target_bcp.shape)}")

        point = nn.functional.smooth_l1_loss(pred_bcp, target_bcp, beta=self.huber_beta, reduction="none")
        if self.event_weight > 0:
            salience = self._event_salience(target_bcp)
            point = point * (1.0 + self.event_weight * salience)
        loss = point.mean()

        if self.deriv_weight > 0 and pred_bcp.shape[-1] > 1:
            pred_d = torch.diff(pred_bcp, dim=-1)
            target_d = torch.diff(target_bcp, dim=-1)
            loss = loss + self.deriv_weight * nn.functional.smooth_l1_loss(pred_d, target_d, beta=self.huber_beta)

        if self.envelope_weight > 0:
            pred_env = _local_rms_torch(pred_bcp, self.envelope_window)
            target_env = _local_rms_torch(target_bcp, self.envelope_window)
            loss = loss + self.envelope_weight * nn.functional.l1_loss(pred_env, target_env)

        if self.band_weight > 0 and pred_bcp.shape[-1] > 2:
            pred_fft = torch.log1p(torch.abs(torch.fft.rfft(pred_bcp, dim=-1)))
            target_fft = torch.log1p(torch.abs(torch.fft.rfft(target_bcp, dim=-1)))
            loss = loss + self.band_weight * nn.functional.l1_loss(pred_fft, target_fft)

        return loss


def build_criterion(args):
    name = args.loss.lower()

    if name == "hybrid":
        return HybridLoss(
            fft_weight=getattr(args, "fft_weight", 0.1),
            deriv_weight=getattr(args, "deriv_weight", 1.0),
            cont_weight=getattr(args, "cont_weight", 5.0),
            phase_weight=getattr(args, "hybrid_phase_weight", 0.0),
        )

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    if name == "huber":
        return nn.SmoothL1Loss(beta=float(getattr(args, "huber_beta", 0.3)))

    if name == "wmse":
        return WeightedMSE(alpha=float(getattr(args, "wmse_alpha", 1.0)))

    if name == "qp_hybrid":
        return QPHybridLoss(
            deriv_weight=getattr(args, "qp_deriv_weight", 0.5),
            envelope_weight=getattr(args, "qp_envelope_weight", 0.5),
            band_weight=getattr(args, "qp_band_weight", 0.05),
            event_weight=getattr(args, "qp_event_weight", 1.0),
            envelope_window=getattr(args, "qp_envelope_window", 9),
            huber_beta=getattr(args, "huber_beta", 0.3),
        )

    raise ValueError(f"Unknown loss: {args.loss}")
