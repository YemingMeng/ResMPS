import torch as th
import torch.nn as nn
from Tools import feature_map, hilbert_path
from collections import deque
from math import log2
import time
import random


class ResMPS(th.nn.Module):
    """ Residual Matrix Product State

    Parameters for specific cases:
    regular MPS: perturbation=False
    Simple ResMPS: perturbation=True, feature_map_funcs=[lambda x:x]
    activated ResMPS: perturbation=True, relu=True, dropout_prob=0.5
    for more imformation please check https://arxiv.org/abs/2012.11841
    """

    def __init__(self, d: int, D: int, n_site: int, n_label: int,
                 std=1e-3,
                 device=None,
                 renorm=False, renorm_grad=False,
                 forward_norm=1,
                 rotation=False,
                 path=None,
                 perturbation=True,
                 relu=False,
                 dropout_prob=0,
                 parallel_forward=True,
                 feature_map_funcs=None,
                 tsne=False
                 ):
        """ d: int, the dimension of physical bonds.
        D: int, the dimension of geometry bonds.
        n_site: int, the total number of tensors of ResMPS.
        n_label: int, the dimension of the label space.
        std: float, the standard deviation of initialization.
        device: str, "cpu", "cuda", "cuda:0", etc.
        remorm: bool, if Ture, calling self.project_tensor
            will renormalize each tensor.
        renorm_grad: Bool, if True, calling self.project_grad
            will renormalize gradient of each tensor.
        froward_norm: int, 1 for regular MPS and 2 for Bayesian MPS
            (see arxiv:1912.12923).
        rotation: Bool, if True, the gradient will project into
            the tangent space of ResMPS, i.e. the l2 norm of each
            tensor will be conserved.
        path: str, use "z" by default, "z" or "normal" for zigzag
            path, "shuffle" for random path and "hilbert" for
            hilbert path. give the mapping that how 2D data be
            flattened into 1D sequence.
        perturbation: bool, if True add residual connection and False
            for regular MPS.
        relu: bool, if True add ReLU activation, only valid when
            perturbation set to be True.
        dropout_prob: float, the probability of dropout layer, only
            valid when perturbation set to be True.
        parallel_forward: bool, if True, use parallel contraction.
        feature_map_funcs: list of function, use [x, 1-x] by default.
        tsne: bool, if True, collect intermediate hidden variables
        """
        super( ).__init__( )
        if device is None:
            device = 'cuda' if th.cuda.is_available( ) else 'cpu'
        if path is None:
            self.path = 'z'
        else:
            self.path = path
        if feature_map_funcs is not None:
            assert len(feature_map_funcs) == d
        self.feature_map = lambda data: feature_map(data, feature_map_funcs)
        self.device = device
        self.renorm = renorm
        self.renorm_grad = renorm_grad
        self.rotation = rotation
        self.forward_norm = forward_norm
        self.perturbation = perturbation
        self.relu = relu
        self.d, self.D, self.n_site, self.n_label = d, D, n_site, n_label
        self.tsne = tsne
        self.ledge = self.random_initializer(1, D, boundary=True).to(device)
        self.redge = self.random_initializer(1, D, boundary=True).to(device)
        lrten = self.random_initializer(
            (n_site-1)*d, D, perturbation=perturbation, std=std).to(device)
        ctten = self.random_initializer(
            n_label, D, perturbation=perturbation, std=std).to(device)
        lrten = lrten.reshape((n_site-1), d, D, D)
        lrten_para = th.nn.Parameter(lrten)
        ctten_para = th.nn.Parameter(ctten)
        self.layers = th.nn.ParameterDict(
            {'lrten': lrten_para,
             'ctten': ctten_para
             })
        if relu:
            assert perturbation
            bias = nn.Parameter(th.zeros(n_site-1, D, device=device))
            self.layers['bias'] = bias
        if dropout_prob > 0:
            assert perturbation
            self.dropout = th.nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None
        self.lrmask = th.ones_like(lrten).to(device)
        self.project_tensor( )
        self._path_seed = time.time( )
        if self.tsne:
            self.forward = self.tsne_forward
        elif parallel_forward:
            self.forward = self.parallel_forward
        else:
            self.forward = self.serial_forward

    def project_grad(self):
        """ if self.ratation, project the gradient to tangent space,
        if self.renorm_grad, renorm the gradient.
        """
        if self.rotation:
            self._rotation( )
        if self.renorm_grad:
            self._renorm_grad( )

    def project_tensor(self):
        """ if self.reorm, apply renormalization to each tensor.
        """
        if self.renorm:
            self._renorm( )

    def parallel_forward(self, data: th.tensor):
        """ Forward Propagation of ResMPS, parallel version.
        """
        data = self.feature_map(data)
        data = self.flatten_data(data, path=self.path)
        if self.forward_norm == 1:
            lrten = self.layers['lrten']
            ctten = self.layers['ctten']
        elif self.forward_norm == 2:
            lrten = self.layers['lrten']**2
            ctten = self.layers['ctten']**2
        lrten = lrten * self.lrmask
        a, b, c, d = data.shape
        if b == 3:
            data = data.view(a, 1, b*c, d)/3
        mats = th.einsum('bcdl, ldxy -> bclxy', data, lrten)
        a, b, c, d, e = mats.shape
        if self.perturbation:
            mats += (th.eye(d, device=self.device)
                     .view(1, 1, 1, d, e).repeat(a, b, c, 1, 1))
        lten = mats[:, :, :self.n_site//2, :, :]
        rten = mats[:, :, self.n_site//2:, :, :]
        lten = self.para_contraction(lten)
        rten = self.para_contraction(rten)
        lten = th.matmul(self.ledge, lten)
        rten = th.matmul(rten, self.redge.t( ))
        rst  = th.einsum("abcd, abef, ode -> abo",
                         lten, rten, ctten)
        return rst

    def serial_forward(self, data: th.tensor):
        """ Forward Propagation of ResMPS, serial version.
        """
        data = self.feature_map(data)
        data = self.flatten_data(data, path=self.path)
        data = deque(th.split(data, 1, dim=-1))
        for idx, i in enumerate(data):
            a, b, c, d = i.shape
            if b == 3:
                data[idx] = i.view(a, b*c)/3
            else:
                data[idx] = i.view(a, b*c)
        lrten = self.layers['lrten'] * self.lrmask
        v1 = self.ledge
        v2 = data.popleft( )
        t  = lrten[0]
        if self.forward_norm == 2:
            t = t**2
        v3 = th.einsum('ol, bd, dlr -> br', v1, v2, t)
        if self.perturbation:
            if self.relu:
                v3 += self.layers['bias'][0, :]
                v3 = nn.functional.relu(v3)
            if self.dropout:
                v3 = self.dropout(v3)
            v3 = v3 + v1.repeat(v3.shape[0], 1)
        data.appendleft(v3)
        v1 = self.redge
        v2 = data.pop( )
        t  = lrten[-1]
        if self.forward_norm == 2:
            t = t**2
        v3 = th.einsum('or, bd, dlr -> bl', v1, v2, t)
        if self.perturbation:
            if self.relu:
                v3 += self.layers['bias'][-1, :]
                v3 = nn.functional.relu(v3)
            if self.dropout:
                v3 = self.dropout(v3)
            v3 = v3 + v1.repeat(v3.shape[0], 1)
        data.append(v3)
        a = 1
        while len(data) > 2:
            v1 = data.popleft( )
            v2 = data.popleft( )
            t  = lrten[a]
            if self.forward_norm == 2:
                t = t**2
            v3 = th.einsum('bl, bd, dlr -> br', v1, v2, t)
            if self.perturbation:
                if self.relu:
                    v3 += self.layers['bias'][a, :]
                    v3 = nn.functional.relu(v3)
                if self.dropout:
                    v3 = self.dropout(v3)
                v3 = v3 + v1
            data.appendleft(v3)
            v1 = data.pop( )
            v2 = data.pop( )
            t  = lrten[-1-a]
            if self.forward_norm == 2:
                t = t**2
            v3 = th.einsum('br, bd, dlr -> bl', v1, v2, t)
            if self.perturbation:
                if self.relu:
                    v3 += self.layers['bias'][-1-a, :]
                    v3 = nn.functional.relu(v3)
                if self.dropout:
                    v3 = self.dropout(v3)
                v3 = v3 + v1
            data.append(v3)
            a += 1
        v1 = data[0]
        v2 = data[1]
        t  = self.layers['ctten']
        if self.forward_norm == 2:
            t = t**2
        rst = th.einsum('bl, br, ulr -> bu', v1, v2, t)
        a, b = rst.shape
        rst  = rst.view(a, 1, b)
        return rst

    def tsne_forward(self, data: th.tensor):
        """ Forward Propagation of ResMPS
            t-SNE version that collects hidden variables
        """
        data = self.feature_map(data)
        data = self.flatten_data(data, path=self.path)
        data = deque(th.split(data, 1, dim=-1))
        for idx, i in enumerate(data):
            a, b, c, d = i.shape
            if b == 3:
                data[idx] = i.view(a, b*c)/3
            else:
                data[idx] = i.view(a, b*c)
        lrten = self.layers['lrten'] * self.lrmask
        v1 = self.ledge
        v2 = data.popleft( )
        t  = lrten[0]
        if self.forward_norm == 2:
            t = t**2
        v3 = th.einsum('ol, bd, dlr -> br', v1, v2, t)
        if self.perturbation:
            if self.relu:
                v3 += self.layers['bias'][0, :]
                v3 = nn.functional.relu(v3)
            if self.dropout:
                v3 = self.dropout(v3)
            v3 = v3 + v1.repeat(v3.shape[0], 1)
        self.hiddens =[v3]
        data.appendleft(v3)
        a = 1
        while len(data) > 1:
            v1 = data.popleft( )
            v2 = data.popleft( )
            t  = lrten[a]
            if self.forward_norm == 2:
                t = t**2
            v3 = th.einsum('bl, bd, dlr -> br', v1, v2, t)
            if self.perturbation:
                if self.relu:
                    v3 += self.layers['bias'][a, :]
                    v3 = nn.functional.relu(v3)
                if self.dropout:
                    v3 = self.dropout(v3)
                v3 = v3 + v1
            self.hiddens.append(v3)
            data.appendleft(v3)
            a += 1
        v1 = data[0]
        v2 = self.redge
        t  = self.layers['ctten']
        if self.forward_norm == 2:
            t = t**2
        rst = th.einsum('bl, or, ulr -> bu', v1, v2, t)
        a, b = rst.shape
        rst  = rst.view(a, 1, b)
        return rst

    def para_contraction(self, ten):
        """ parallel contraction of MPS
        """
        d0, d1, size, d3, d4 = ten.shape
        while size > 1:
            half_size = size // 2
            nice_size = half_size * 2
            leftover  = ten[:, :, nice_size:, :, :]
            odd = ten[:, :, 0:nice_size:2, :, :]
            eve = ten[:, :, 1:nice_size:2, :, :]
            ten = th.matmul(odd, eve)
            ten = th.cat((ten, leftover), 2)
            _, _, size, _, _ = ten.shape
        return ten.view(d0, d1, d3, d4)

    def prune(self, critical_value, mode='percent'):
        """ prune ResMPS, set elements with minor absolute values to 0.
        """
        print('\nstarting pruning process...')
        print('n1 =', th.sum(self.lrmask).item( ))
        with th.no_grad( ):
            if mode == 'percent':
                temp = th.abs(self.layers['lrten'] * self.lrmask).view(-1)
                num = int(len(temp) * critical_value)
                position = temp.topk(num, largest=False)[1]
                self.lrmask.view(-1)[position] = 0
            elif mode == 'magnetude':
                position = th.abs(self.layers['lrten']) < critical_value
                self.lrmask[position] = 0
            else:
                raise ValueError(f"mode {mode} undefined!")
        print('n2 =', th.sum(self.lrmask).item( ))

    def _rotation(self):
        """ project the gradient to tangent space.
        """
        lq = self.layers['lrten'].data[:self.n_site//2, :, :, :]
        rq = self.layers['lrten'].data[self.n_site//2:, :, :, :]
        cq = self.layers['ctten'].data
        lg = self.layers['lrten'].grad.data[:self.n_site//2, :, :, :]
        rg = self.layers['lrten'].grad.data[self.n_site//2:, :, :, :]
        cg = self.layers['ctten'].grad.data
        ltemp = th.einsum('ndlr, ndlr -> ndl ', lg, lq)
        ltemp = th.einsum('ndlr, ndl  -> ndlr', lq, ltemp)
        rtemp = th.einsum('ndlr, ndlr -> ndr ', rg, rq)
        rtemp = th.einsum('ndlr, ndr  -> ndlr', rq, rtemp)
        ctemp = th.einsum('ulr , ulr  -> lr  ', cg, cq)
        ctemp = th.einsum('ulr , lr   -> ulr ', cq, ctemp)
        lg = lg - ltemp
        rg = rg - rtemp
        cg = cg - ctemp
        self.layers['lrten'].grad.data = th.cat((lg, rg), 0)
        self.layers['ctten'].grad.data = cg

    def _renorm_grad(self):
        lg = self.layers['lrten'].grad.data[:self.n_site//2, :, :, :]
        rg = self.layers['lrten'].grad.data[self.n_site//2:, :, :, :]
        cg = self.layers['ctten'].grad.data
        lnorm = th.norm(lg, p=self.forward_norm, dim=3)
        rnorm = th.norm(rg, p=self.forward_norm, dim=2)
        cnorm = th.norm(cg, p=self.forward_norm, dim=0)
        lg = th.einsum("ijkl, ijk -> ijkl", lg, 1/(lnorm+1e-15))
        rg = th.einsum("ijkl, ijl -> ijkl", rg, 1/(rnorm+1e-15))
        cg = th.einsum("ijk ,  jk -> ijk ", cg, 1/(cnorm+1e-15))
        self.layers['lrten'].grad.data = th.cat((lg, rg), 0)
        self.layers['ctten'].grad.data = cg

    def _renorm(self):
        """ apply renormalization to each tensor,
        """
        lten  = self.layers['lrten'].data[:self.n_site//2, :, :, :]
        rten  = self.layers['lrten'].data[self.n_site//2:, :, :, :]
        cten  = self.layers['ctten'].data
        lnorm = th.norm(lten, p=self.forward_norm, dim=3)
        rnorm = th.norm(rten, p=self.forward_norm, dim=2)
        cnorm = th.norm(cten, p=self.forward_norm, dim=0)
        lten  = th.einsum("ijkl, ijk -> ijkl", lten, 1/(lnorm+1e-15))
        rten  = th.einsum("ijkl, ijl -> ijkl", rten, 1/(rnorm+1e-15))
        cten  = th.einsum("ijk ,  jk -> ijk ", cten, 1/(cnorm+1e-15))
        self.layers['lrten'].data = th.cat((lten, rten), 0)
        self.layers['ctten'].data = cten

    def flatten_data(self, data, path='z'):
        """ transform 2D data into 1D sequence according to path.
        """
        if path == 'z' or path == 'normal':
            batch_size, nchannel, d, row, col = data.shape
            rst = data.view(batch_size, nchannel, d, -1)
        elif path == 'shuffle' or path == 'random':
            batch_size, nchannel, d, row, col = data.shape
            path_list = list(range(self.n_site-1))
            random.Random(self._path_seed).shuffle(path_list)
            rst = data.view(batch_size, nchannel, d, -1)[:, :, :, path_list]
        elif path == 'hilbert':
            n = int(log2(self.n_site-1))//2
            x, y = hilbert_path(n)
            rst = data[:, :, :, x, y]
        assert rst.shape[-1] == self.n_site - 1
        return rst

    @staticmethod
    def random_initializer(d: int, D: int, std: float = 1e-3,
                           boundary: bool = False,
                           perturbation: bool = False) -> th.nn.Module:
        """ use Gaussian noise to initialize MPS.
        d: int, the dimension of physical bonds.
        D: int, the dimension of geometry bonds.
        std: float, the standard deviation of initialization.
        boundary: bool, if True, return a one-hot tensor resides at the
            end of MPS. if False, return bulk tensors.
        perturbation: bool, True for the residual mode, and False for
            the regular MPS mode.
        """
        if boundary:
            x = th.zeros(d, D)
            x[:, 0] = 1
        elif perturbation:
            x = th.clamp(th.randn(d, D, D) * std, min=-2*std, max=2*std)
        else:
            x = th.eye(D).reshape(1, D, D)
            x = x.repeat(d, 1, 1)
            x += th.clamp(th.randn(x.shape) * std, min=-2*std, max=2*std)
        return x


def example( ):
    cf = ResMPS(2, 12, 785, 10)
    input = th.rand(1000, 1, 28, 28, device=cf.device)
    output = cf(input)
    print(output.shape)


if __name__ == "__main__":
    example( )
