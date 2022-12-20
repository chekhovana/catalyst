import os.path
from typing import Iterable, Dict, Union, Optional

import wandb
import catalyst
from catalyst import utils
from catalyst.settings import SETTINGS
from catalyst.loggers import WandbLogger


class CustomWandbLogger(WandbLogger):
    def __init__(self, project: str, name: Optional[str] = None,
                 entity: Optional[str] = None,
                 log_batch_metrics: bool = SETTINGS.log_batch_metrics,
                 log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
                 configs: Iterable[str] = None,
                 files: Iterable[str] = None,
                 **kwargs) -> None:
        super().__init__(project, name, entity, log_batch_metrics,
                         log_epoch_metrics, **kwargs)
        self.artifact = wandb.Artifact(f'{self.run.name}', type='artifact')

        api = wandb.Api()
        self.apirun = api.run(
            f'{self.run.entity}/{self.run.project}/{self.run.id}')

        self.configs = configs
        if configs is not None:
            parts = [utils.load_config(cp) for cp in configs]
            config = utils.merge_dicts(*parts) if len(parts) > 1 else parts[0]
            self.run.config.update(config)
        self.files = files
        self.runner = None

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        # super().log_hparams(hparams, runner)
        if self.configs is not None:
            self.log_artifact('config', runner, path_to_artifact=self.configs)
        self.runner = runner

    # def log_metrics(self, metrics: Dict[str, float], scope: str,
    #                 runner: "IRunner") -> None:
    #     super().log_metrics(metrics, scope, runner)
    #     if scope != 'epoch':
    #         return
    #     self.remove_artifacts('checkpoints')
    #     self.log_artifact('checkpoints', runner, path_to_artifact=self.files)
    #     # self.remove_artifacts_without_aliases()

    # def remove_artifacts_without_aliases(self):
    #     for a in self.apirun.logged_artifacts():
    #         if len(a.aliases) == 0:
    #             # if alias in a.aliases:
    #             a.delete(delete_aliases=True)
    #             a.wait()

    # def remove_artifacts(self, alias):
    #     for a in self.apirun.logged_artifacts():
    #         if alias in a.aliases:
    #             # if alias in a.aliases:
    #             a.delete(delete_aliases=True)
    #             a.wait()

    def log_artifact(self, tag: str, runner: "IRunner", artifact: object = None,
                     path_to_artifact: Union[str, list] = None,
                     scope: str = None) -> None:
        artifact = wandb.Artifact(f'{self.run.name}', type='artifact')
        if isinstance(path_to_artifact, list):
            [artifact.add_file(p) for p in path_to_artifact]
        elif os.path.isdir(path_to_artifact):
            artifact.add_dir(path_to_artifact)
        else:
            artifact.add_file(path_to_artifact)
        self.run.log_artifact(artifact, aliases=[tag])

    def close_log(self, scope: str = None) -> None:
        self.log_artifact('checkpoints', self.runner,
                          path_to_artifact=self.files)

        """Closes the logger."""
        self.run.finish()
