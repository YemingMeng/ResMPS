import torch as th
from hilbertcurve.hilbertcurve import HilbertCurve
from time import time
from operator import mul
from functools import reduce


def label2onehot(labels, select=None):
    if select is None:
        nlabel = 10
        nlabels = len(labels)
        onehot = th.zeros(nlabels, nlabel, device=labels.device)
        for i in range(nlabels):
            onehot[i, labels[i]] = 1
    else:
        nlabel = len(select)
        nlabels = len(labels)
        onehot = th.zeros(nlabels, nlabel, device=labels.device)
        num2idx = dict( )
        for idx, num in enumerate(select):
            num2idx[num] = idx
        for i in range(nlabels):
            num = int(labels[i])
            onehot[i, num2idx[num]] = 1
    return onehot


def expand_data(data, L=32):
    *a, b, c = data.shape
    ndata = th.zeros(*a, L, L)
    b1 = (L-b)//2
    b2 = (L-b)//2 + b
    c1 = (L-c)//2
    c2 = (L-c)//2 + c
    string = ("ndata[{}, b1:b2, c1:c2] = data"
              .format(', '.join([':' for i in a])))
    _locals = locals()
    exec(string, _locals)
    return ndata


def feature_map(data, funcs=None):
    if funcs is None:
        funcs = (lambda x: x, lambda x: 1-x)
    return th.stack([f(data) for f in funcs], dim=-3)


def nll(predict, target):
    batch_size, _ = predict.shape
    temp = -(target*th.log(predict))/batch_size
    return th.sum(temp)


def projection(parameters, rotation=True, renorm=True):
    for q in parameters:
        g = q.grad.data
        with th.no_grad( ):
            ncdim = g.dim( )-1
            lettlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            cstr = ''.join(lettlist[:g.ndim-1])
            str1 = '{}k, {}k -> {} '.format(cstr, cstr, cstr)
            str2 = '{}k, {}  -> {}k'.format(cstr, cstr, cstr)
            if rotation:
                # projection the gradient to the tangent space
                temp = th.einsum(str1, g, q)
                temp = th.einsum(str2, q, temp)
                g = g - temp

            # renorm gradient
            if renorm:
                temp = 1 / (th.sqrt(th.sum(g**2, axis=ncdim))+1e-14)
                g = th.einsum(str2, g, temp)
        q.grad.data = g
    return 0


def hilbert_path(n):
    hilbert_curve = HilbertCurve(n, 2)
    lx = [ ]
    ly = [ ]
    for i in range(2**(n*2)):
        try:
            x, y = hilbert_curve.coordinates_from_distance(i)
        except AttributeError:
            x, y = hilbert_curve.point_from_distance(i)
        lx.append(x)
        ly.append(y)
    tx, ty = tuple(lx), tuple(ly)
    return tx, ty


def show_path(n):
    import matplotlib.pyplot as pl
    x, y = hilbert_path(n)
    pl.figure( )
    for i in range(len(x)-1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]
        pl.plot([x1, x2], [y1, y2], 'b')
        pl.axis('equal')
    pl.tight_layout( )
    pl.show( )


def timer(func):
    def wrapper(*args, **kw):
        t1 = time( )
        rst = func(*args, **kw)
        dt = time( ) - t1
        print(f'run {func.__name__} in {dt:.2f} seconds')
        return rst
    return wrapper


def trid(D, dim=2, device=None):
    if device is None:
        device = 'cuda' if th.cuda.is_available( ) else 'cpu'
    indexs = [th.arange(D, device=device) for i in range(dim)]
    Indexs = th.meshgrid(indexs)
    temp   = (i > j for i, j in zip(Indexs[:-1], Indexs[1:]))
    temp   = reduce(mul, temp)
    temp   = temp.float( )
    return temp
