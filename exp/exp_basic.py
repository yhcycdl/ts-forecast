
import torch
import torch.nn as nn

# 导入“模型模块”（注意：是模块，不是类）
#from models import full_res_tcn
from models import cnn_lstm , CRNN, GRU, fast_tcn ,Fullrestcn,spectral,inceptiontime,Dlinear,PatchTST,mamba,timemixer,tcn_claude,timemixer_claude,risk_cnn,smooth_pecnet
import os

class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        # TimeMixer 风格：model_dict 映射到“模块”
        self.model_dict = {
            "CNNLSTM": cnn_lstm,
            "CRNN": CRNN,
            "GRU" : GRU,
            "fast_tcn":fast_tcn,
            "Fullrestcn":Fullrestcn,
            "spetical":spectral,
            "inceptiontime": inceptiontime,
            "DLinear":Dlinear,
            "PatchTST":PatchTST,
            "mamba":mamba,
            "timemixer":timemixer,
            "tcn_claude":tcn_claude,
            "smooth_pecnet": smooth_pecnet,
            "timemixer_claude":timemixer_claude,
            "risk_cnn": risk_cnn,

        }

        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if getattr(self.args, "use_gpu", False) and torch.cuda.is_available():
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")

    def _build_model(self):
        if self.args.model not in self.model_dict:
            raise ValueError(f"Unknown model: {self.args.model}. "
                             f"Available: {list(self.model_dict.keys())}")

        # 关键：每个模型模块都必须提供 Model(args)
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
