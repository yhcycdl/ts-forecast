from models import inceptiontime
from models.joint_wrapper_base import JointForecastRiskWrapper


class Model(JointForecastRiskWrapper):
    def __init__(self, configs):
        super().__init__(inceptiontime.Model(configs))
