from torch import nn
import copy


class YatzyNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_dim, bias=True)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)
