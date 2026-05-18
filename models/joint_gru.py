from models import GRU
from models.joint_wrapper_base import JointForecastRiskWrapper


class Model(JointForecastRiskWrapper):
    def __init__(self, configs):
        super().__init__(GRU.Model(configs))
