import torch as th
import torch.nn as nn
from Tools import trid


class EM(th.nn.Module):

    def __init__(self, D: int, n_res: int, n_class: int,
                 relu=False,
                 dropout_prob=0,
                 device=None,
                 order=None,
                 std=1e-3):
        super( ).__init__( )
        self.D, self.n_res, self.n_class = D, n_res, n_class
        self.relu  = relu
        if order is None:
            order = 1
        self.order = order
        if device is None:
            device = 'cuda' if th.cuda.is_available( ) else 'cpu'
        self.device      = device
        self.lrten       = nn.Parameter(
            th.randn(n_res, D, D, device=device) * std)
        self.vir_feature = th.ones(1, D, device=device) / D
        self.bias        = nn.Parameter(
            th.zeros(n_res, D, device=device))
        self.fc      = nn.Linear(D, n_class).to(device)
        if dropout_prob > 0:
            self.dropout = th.nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None
        if order == 2:
            self.M  = trid(n_res, dim=2, device=device)
        else:
            self.M1 = trid(n_res, dim=2, device=device)
            self.M2 = trid(n_res, dim=3, device=device)

    def forward(self, data: th.tensor):
        vf   = th.squeeze(self.vir_feature)
        data = data.view(*data.shape[:-2], -1).squeeze(1)
        if self.order == 1:
            if not self.relu:
                temp = th.einsum('nlr, bn -> blr', self.lrten, data)
                temp = th.einsum('l, blr -> br', vf, temp)
            else:
                temp = th.einsum('nlr, bn -> bnlr', self.lrten, data)
                temp = th.einsum('l, bnlr -> bnr', vf, temp)
                temp = temp + self.bias
                temp = nn.functional.relu(temp)
                if self.dropout:
                    temp = self.dropout(temp)
                temp = th.sum(temp, axis=1)
        if self.order == 2:
            temp1 = th.einsum('nlr, bn -> blr', self.lrten, data)
            temp1 = th.einsum('l, blr -> br', vf, temp1)

            W    = th.einsum('nlr, bn -> bnlr', self.lrten, data)
            lW   = th.einsum('l, bnlr -> bnr', vf, W)
            lWM  = th.einsum('bnr, nm -> bmr', lW, self.M)
            temp2 = th.einsum('bmr, bmro -> bo', lWM, W)

            temp = temp1 + temp2
        if self.order == 3:
            temp1 = th.einsum('nlr, bn -> blr', self.lrten, data)
            temp1 = th.einsum('l, blr -> br', vf, temp1)

            W     = th.einsum('nlr, bn -> bnlr', self.lrten, data)
            lW    = th.einsum('l, bnlr -> bnr', vf, W)
            lWM   = th.einsum('bnr, nm -> bmr', lW, self.M1)
            temp2 = th.einsum('bmr, bmro -> bo', lWM, W)

            lW0W1 = th.einsum('bix, bjxy -> bijy', lW, W)
            lW0W1 = th.einsum('bijy, ijk -> bky', lW0W1, self.M2)
            temp3 = th.einsum('bky, bkyo -> bo', lW0W1, W)

            temp = temp1 + temp2 + temp3
        vf   = temp + vf
        rst  = self.fc(vf)
        return rst

    def project_grad(self):
        pass

    def project_tensor(self):
        pass


class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__( )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU( ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU( ),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


class FullLinear(th.nn.Module):

    def __init__(self, n_res: int, n_class: int,
                 device=None, depth=1):
        super( ).__init__( )
        self.n_res, self.n_class = n_res, n_class
        if device is None:
            device = 'cuda' if th.cuda.is_available( ) else 'cpu'
        self.device = device
        self.depth = depth
        if depth == 1:
            self.Linear = th.nn.Linear(n_res, n_class).to(device)
        elif depth == 2:
            self.Linear1 = th.nn.Linear(n_res, n_res).to(device)
            self.Linear2 = th.nn.Linear(n_res, n_class).to(device)
        elif depth == 3:
            self.Linear1 = th.nn.Linear(n_res, n_res).to(device)
            self.Linear2 = th.nn.Linear(n_res, n_res).to(device)
            self.Linear3 = th.nn.Linear(n_res, n_class).to(device)

    def forward(self, data: th.tensor):
        data = data.view(*data.shape[:-2], -1).squeeze(1)
        if self.depth == 1:
            rst  = self.Linear(data)
        elif self.depth == 2:
            data = self.Linear1(data)
            data = nn.functional.relu(data)
            rst  = self.Linear2(data)
        elif self.depth == 3:
            data = self.Linear1(data)
            data = nn.functional.relu(data)
            data = self.Linear2(data)
            data = nn.functional.relu(data)
            rst  = self.Linear3(data)
        return rst

    def project_grad(self):
        pass

    def project_tensor(self):
        pass


def main( ):
    cf = FullLinear(784, 10)
    data = th.rand(1000, 1, 28, 28, device=cf.device)
    print(cf(data).shape)


if __name__ == "__main__":
    main( )
