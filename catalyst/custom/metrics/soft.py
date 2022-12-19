from typing import Optional
from typing import Tuple
import numpy as np
import torch

from catalyst.metrics import BinaryPrecisionRecallF1Metric


class SoftBinaryMetric(BinaryPrecisionRecallF1Metric):

    def __init__(self, zero_division: int = 0, compute_on_call: bool = True,
                 prefix: Optional[str] = None, suffix: Optional[str] = None):
        super().__init__(zero_division, compute_on_call, prefix, suffix)
        self.outputs, self.targets = None, None

    def reset(self) -> None:
        super().reset()
        self.outputs = np.zeros((0, 1))
        self.targets = np.zeros((0, 1))

    def calculate(self, outputs, targets) -> Tuple[float, float, float]:
        beta = 1
        outputs = outputs.clip(0, 1)
        y_true_count = targets.sum()
        tp = outputs[targets == 1].sum()
        fp = outputs[targets == 0].sum()
        fn = y_true_count - tp
        beta_squared = beta * beta
        c_precision = tp / (tp + fp)
        c_recall = tp / y_true_count
        c_f1 = 0
        if c_precision > 0 and c_recall > 0:
            c_f1 = (1 + beta_squared) * (c_precision * c_recall) / (
                        beta_squared * c_precision + c_recall)
        return c_precision, c_recall, c_f1

    def update(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float, float]:
        outputs = outputs.clip(0, 1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        self.outputs = np.vstack((self.outputs, outputs))
        self.targets = np.vstack((self.targets, targets))
        return self.calculate(self.outputs, self.targets)

    def compute(self) -> Tuple[float, float, float]:
        result = self.calculate(self.outputs, self.targets)
        return result
