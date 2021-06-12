# %% md
import torch
from examples.notebooks.utils import accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.oneshot.pytorch import DartsTrainer
import itertools
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from collections import OrderedDict

# Retiarii Example - One-shot NAS

# %% md
#
# This
# example
# will
# show
# Retiarii
# 's ability to **express** and **explore** the model space for Neural Architecture Search and Hyper-Parameter Tuning in a simple way. The video demo is in [YouTube](https://youtu.be/3nEx9GMHYEk) and [Bilibili](https://www.bilibili.com/video/BV1c54y1V7vx/).
#
# Let
# 's start the journey with Retiarii!

# %% md

## Step 1: Express the Model Space

### Step 1.1: Define the Base Model

class CIFAR_17(nn.Module):
    '''
    BaseModel which has 3 CNN layers and 2 FC layers
    '''

    def __init__(self, head_size=10):
        super(CIFAR_17, self).__init__()

        self.body = nn.Sequential(OrderedDict([
            ('cnn1', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(3, 8, 3, 1, 1)),
                ('batchnorm', nn.BatchNorm2d(8)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(2))
            ]))),
            ('cnn2', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(8, 8, 3, 1, 1)),
                ('batchnorm', nn.BatchNorm2d(8)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(2))
            ]))),
            ('cnn3', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(8, 8, 3, 1, 1)),
                ('batchnorm', nn.BatchNorm2d(8)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(2)),
            ])))
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('dense', nn.Sequential(OrderedDict([
                ('fc1', nn.Conv2d(8 * 4 * 4, 32, kernel_size=1, bias=True)),  # implement dense layer in CNN way
                ('relu', nn.ReLU(inplace=True)),
                ('fc2', nn.Conv2d(32, head_size, kernel_size=1, bias=True)),
            ])))
        ]))

    def features(self, x):
        feat = self.body(x)
        feat = x.view(x.shape[0], -1)
        return feat

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.shape[0], -1, 1, 1)  # flatten
        x = self.head(x)
        x = x.view(x.shape[0], -1)
        return x

model = CIFAR_17()

# %% md

### Step 1.2: Define the Model Mutations

# %%

import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn


class Net(nn.Module):
    def __init__(self, head_size=10, lower_range=8, upper_range=64):
        super(Net, self).__init__()
        self.head_size = head_size
        self.lower_range = lower_range
        self.upper_range = upper_range
        choice_dict = self._get_mutator()
        self.net = nn.LayerChoice(choice_dict)

    def _get_mutator(self):
        layer_choices = []
        a = [range(self.lower_range, self.upper_range+1),
             range(self.lower_range, self.upper_range+1),
             range(self.lower_range, self.upper_range+1)]

        for comb in list(itertools.product(*a)):
            i, j, k = comb
            layer_choices.append(
                nn.Sequential(
                        nn.Conv2d(3, i, 3, 1, 1),
                        nn.BatchNorm2d(i),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),

                        nn.Conv2d(i, j, 3, 1, 1),
                        nn.BatchNorm2d(j),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),

                        nn.Conv2d(j, k, 3, 1, 1),
                        nn.BatchNorm2d(k),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),

                        nn.AdaptiveAvgPool2d((4, 4)),
                        nn.Linear(k * 4 * 4, 32,  bias=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(32, self.head_size,  bias=True),
                    )
            )
        return layer_choices

    def forward(self, x):
        out = self.net(x)
        return out


model = Net()
model.forward(torch.rand(1, 3, 32, 32))


## Step 2: Explore the Model Space

# %%


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

train_dataset = CIFAR10(root="./data",
                        train=True,
                        download=True,
                        transform=transform)

trainer = DartsTrainer(
    model=model,
    loss=criterion,
    metrics=lambda output, target: accuracy(output, target),
    optimizer=optimizer,
    num_epochs=2,
    dataset=train_dataset,
    batch_size=256,
    log_frequency=10,
)

trainer.fit()

# %% md
#
# Similarly, the optimal structure found can be exported.

# %%

print('Final architecture:', trainer.export())

# %%


