import os

import torch
import torch.nn as nn

from models import (
    joint_cnn_lstm,
    joint_crnn,
    joint_dlinear,
    joint_fast_tcn,
    joint_fullrestcn,
    joint_gru,
    joint_inceptiontime,
    joint_mamba,
    joint_patchtst,
    joint_risk_cnn,
    joint_spectral,
    joint_tcn_claude,
    joint_timemixer,
    joint_timemixer_claude,
)


class Exp_Basic_Joint:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        self.model_dict = {
            "joint_cnn_lstm": joint_cnn_lstm,
            "joint_crnn": joint_crnn,
            "joint_gru": joint_gru,
            "joint_fast_tcn": joint_fast_tcn,
            "joint_fullrestcn": joint_fullrestcn,
            "joint_spectral": joint_spectral,
            "joint_inceptiontime": joint_inceptiontime,
            "joint_dlinear": joint_dlinear,
            "joint_patchtst": joint_patchtst,
            "joint_mamba": joint_mamba,
            "joint_timemixer": joint_timemixer,
            "joint_tcn_claude": joint_tcn_claude,
            "joint_timemixer_claude": joint_timemixer_claude,
            "joint_risk_cnn": joint_risk_cnn,
        }

        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if getattr(self.args, "use_gpu", False) and torch.cuda.is_available():
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")

    def _build_model(self):
        if self.args.model not in self.model_dict:
            raise ValueError(f"Unknown joint model: {self.args.model}. Available: {list(self.model_dict.keys())}")

        model = self.model_dict[self.args.model].Model(self.args).float()
        if getattr(self.args, "use_multi_gpu", False) and getattr(self.args, "use_gpu", False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _make_ckpt_dir(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_data(self, flag):
        raise NotImplementedError

    def train(self, setting):
        raise NotImplementedError

    def test(self, setting, test=0):
        raise NotImplementedError
