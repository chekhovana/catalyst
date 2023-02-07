import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
# import catalyst.loggers
from catalyst import dl
from catalyst.custom.loggers import CometLogger

np.random.seed(42)
torch.manual_seed(42)

num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

# pytorch loaders
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)  # , num_workers=0)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

runner = dl.SupervisedRunner(model=model)

logdir = 'logs'
checkpoint_dir = 'logs/checkpoints'
loggers = dict(
    comet=CometLogger(project_name='minimal_example',
                      checkpoint_dir=checkpoint_dir, config_file='setup.py'))
# loggers = dict()
callbacks = [
    dl.AccuracyCallback(input_key="logits", target_key="targets",
                        num_classes=num_classes),
    # dl.PrecisionRecallF1SupportCallback(
    #     input_key="logits", target_key="targets", num_classes=num_classes
    # ),
    # dl.AUCCallback(input_key="logits", target_key="targets"),
    # catalyst[ml] required ``pip install catalyst[ml]``
    # dl.ConfusionMatrixCallback(
    #     input_key="logits", target_key="targets", num_classes=num_classes
    # ),
    dl.CheckpointCallback(logdir=checkpoint_dir, loader_key='valid',
                          metric_key='accuracy01', minimize=False)
]

# runner.train(
#     criterion=criterion,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     loaders=loaders,
#     callbacks=callbacks,
#     logdir=logdir,
#     num_epochs=20,
#     verbose=False,
#     loggers=loggers
# )

print('evaluate')
model.load_state_dict(torch.load('logs/checkpoints/comet.pth'))
# model.load_state_dict(torch.load('/Users/chekhovana/Yandex.Disk.localized/downloads/model.best.pth'))
runner.evaluate_loader(loaders['valid'], callbacks=[
    dl.AccuracyCallback(input_key="logits", target_key="targets",
                        num_classes=num_classes), ]
                       )
