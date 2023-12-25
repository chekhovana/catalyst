import os
from typing import Dict, Optional, List

import catalyst.loggers
from catalyst.settings import SETTINGS
import yaml

from collections.abc import MutableMapping


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
        if config_file:
            self.experiment.log_asset(config_file)
            with open(config_file) as f:
                config = yaml.load(f, Loader=yaml.Loader)
                config = flatten_dict(config)
                self.experiment.log_parameters(config)

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

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        if len(hparams) > 0:
            super().log_hparams(hparams, runner)

    def close_log(self) -> None:
        self.log_model()
        super().close_log()


