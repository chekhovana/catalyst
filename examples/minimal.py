import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl

# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

# pytorch loaders
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=0)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

# model training
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets",
    loss_key="loss"
)
import os
from catalyst.custom.loggers import CustomWandbLogger

logdir = './logs'
files = [os.path.join(logdir, f'checkpoints/model.{cp}.pth') for cp in
         ('last', 'best')]
loggers = dict(wandb=CustomWandbLogger(project='test_catalyst',
                                       files=files))

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir="./logs",
    num_epochs=3,
    valid_loader="valid",
    valid_metric="accuracy03",
    minimize_valid_metric=False,
    verbose=True,
    callbacks=[
        dl.AccuracyCallback(input_key="logits", target_key="targets",
                            num_classes=num_classes),
        # uncomment for extra metrics:
        # dl.PrecisionRecallF1SupportCallback(
        #     input_key="logits", target_key="targets", num_classes=num_classes
        # ),
        # dl.AUCCallback(input_key="logits", target_key="targets"),
        # catalyst[ml] required ``pip install catalyst[ml]``
        # dl.ConfusionMatrixCallback(
        #     input_key="logits", target_key="targets", num_classes=num_classes
        # ),
    ],
    loggers=loggers
)
