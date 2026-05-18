from models import risk_cnn
from models.joint_wrapper_base import JointForecastRiskWrapper


class Model(JointForecastRiskWrapper):
    def __init__(self, configs):
        super().__init__(risk_cnn.Model(configs))
