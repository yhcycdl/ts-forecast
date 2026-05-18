import torch.nn as nn


class JointForecastRiskWrapper(nn.Module):
    """
    给已有的双头模型提供联合任务 forward：
      return {
          "forecast": ...,
          "classification": ...
      }

    这样不需要改原模型文件，只增加新的 joint 入口。
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forecast(self, x):
        return self.base_model.forecast(x)

    def classification(self, x):
        return self.base_model.classification(x)

    def forward(self, x):
        return {
            "forecast": self.forecast(x),
            "classification": self.classification(x),
        }
