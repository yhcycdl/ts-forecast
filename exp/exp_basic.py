
import importlib
import os

import torch
import torch.nn as nn


MODEL_REGISTRY = {
    # Main quasi-periodic waveform models and baselines.
    "tcn_claude": "models.tcn_claude",
    "smooth_pecnet": "models.smooth_pecnet",
    "DLinear": "models.Dlinear",
    "PatchTST": "models.PatchTST",
}


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if getattr(self.args, "use_gpu", False) and torch.cuda.is_available():
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")

    def _build_model(self):
        if self.args.model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {self.args.model}. "
                             f"Available: {list(MODEL_REGISTRY.keys())}")

        # Each model module must expose Model(args). Import lazily so optional
        # baselines do not make the main forecasting path depend on them.
        module = importlib.import_module(MODEL_REGISTRY[self.args.model])
        model = module.Model(self.args).float()

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
