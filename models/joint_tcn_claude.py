from models import tcn_claude
from models.joint_wrapper_base import JointForecastRiskWrapper


class Model(JointForecastRiskWrapper):
    def __init__(self, configs):
        super().__init__(tcn_claude.Model(configs))
