
# %% md
import torch
# from examples.notebooks.utils import accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.oneshot.pytorch import DartsTrainer, EnasTrainer
import itertools
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from collections import OrderedDict

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


def reward_accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size

### Step 1.2: Define the Model Mutations

# %%

import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn


### Step 1.2: Define the Model Mutations


import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn


class Net(nn.Module):
    def __init__(self, head_size=10, lower_range=8, upper_range=16):
        super(Net, self).__init__()
        self.head_size = head_size
        self.lower_range = lower_range
        self.upper_range = upper_range
        choice_dict = self._get_mutator()
        self.net = nn.LayerChoice(choice_dict)

    def _get_mutator(self):
        ## this is supposed to be slooow
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

                        nn.Flatten(),
                        nn.Linear(k * 4 * 4, 32),
                        nn.ReLU(inplace=True),
                        nn.Linear(32, self.head_size),
                    )
            )
        return layer_choices

    def forward(self, x):
        out = self.net(x)
        return out


model = Net()
model.forward(torch.rand(1, 3, 32, 32))



# %% md

## Step 2: Explore the Model Space


import torch
# from nni.retiarii.utils import accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.oneshot.pytorch import DartsTrainer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

trainer = EnasTrainer(
    model=model,
    loss=criterion,
    metrics=accuracy,
    reward_function=reward_accuracy,
    optimizer=optimizer,
    num_epochs=2,
    dataset=train_dataset,
    batch_size=8,
    log_frequency=10,

    #         device=torch.device("cpu")

)

trainer.fit()


print('Final architecture:', trainer.export())

