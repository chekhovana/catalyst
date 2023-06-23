import os
from typing import Dict, Optional, List

import catalyst.loggers
from catalyst.settings import SETTINGS


class CometLogger(catalyst.loggers.CometLogger):

    def __init__(self, workspace: Optional[str] = None,
                 project_name: Optional[str] = None,
                 experiment_id: Optional[str] = None,
                 comet_mode: str = "online", tags: List[str] = None,
                 logging_frequency: int = 1,
                 log_batch_metrics: bool = SETTINGS.log_batch_metrics,
                 log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
                 checkpoint_dir: str = None, config_file: str = None,
                 **experiment_kwargs: Dict) -> None:
        super().__init__(workspace, project_name, experiment_id, comet_mode,
                         tags, logging_frequency, log_batch_metrics,
                         log_epoch_metrics, **experiment_kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.experiment.log_asset(config_file)

    def log_metrics(self, metrics: Dict[str, float], scope: str,
                    runner: "IRunner") -> None:
        if scope == 'epoch':
            for key, value in metrics.items():
                if key.startswith('_'):
                    continue
                self.experiment.log_metrics(
                    value,
                    step=runner.epoch_step,
                    epoch=runner.epoch_step,
                    prefix=f"{key}"
                )
            # self.log_model()

    def log_model(self):
        if self.checkpoint_dir is None:
            return
        for name in ('last', 'best'):
            model_path = os.path.join(self.checkpoint_dir, f'model.{name}.pth')
            self.experiment.log_model(name, file_or_folder=model_path,
                                      overwrite=True)

    def close_log(self) -> None:
        self.log_model()
        super().close_log()


