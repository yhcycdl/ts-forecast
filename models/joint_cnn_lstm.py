from models import cnn_lstm
from models.joint_wrapper_base import JointForecastRiskWrapper


class Model(JointForecastRiskWrapper):
    def __init__(self, configs):
        super().__init__(cnn_lstm.Model(configs))
