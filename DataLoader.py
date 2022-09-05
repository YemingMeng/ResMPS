from Tools import label2onehot, expand_data
import torch as th
import torchvision


class Squeeze( ):
    r""" a class wrap of the function 'torch.squeeze'
    """

    def __call__(self, tensor):
        r""" returns a tensor with all the dimensions of input of size 1 removed.
        """
        return th.squeeze(tensor)


class Loader(object):
    def __init__(self, root="datasets", dataset='mnist', train=True,
                 batch_size=None, shuffle=True, seed=None, squeeze=False,
                 select=None):
        if shuffle:
            if seed is None:
                import time
                seed = time.time( )
            th.manual_seed(seed)
        if batch_size is None:
            if train:
                batch_size = 60000
            else:
                batch_size = 10000

        self.dataset    = dataset
        self.Train      = train
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self.select     = select
        if squeeze:
            tran = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                Squeeze()])
        else:
            tran = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        if self.dataset == 'mnist':
            temp = torchvision.datasets.MNIST(
                root, train=train,
                download=True,
                transform=tran)
        elif self.dataset == 'fashion_mnist':
            temp = torchvision.datasets.FashionMNIST(
                root, train=train,
                download=True,
                transform=tran)
        elif self.dataset == 'cifar10':
            temp = torchvision.datasets.CIFAR10(
                root, train=train,
                download=True,
                transform=tran)
        else:
            raise ValueError("dataset out of range!")
        self.dl = th.utils.data.DataLoader(
            temp, batch_size=batch_size, shuffle=shuffle)
        self.dl = enumerate(self.dl)

    def __iter__(self):
        while True:
            try:
                temp = next(self.dl)
            except StopIteration:
                break
            if self.select is None:
                yield temp
            else:
                yield self.filt(temp)

    def __next__(self):
        if self.select:
            return self.filt(next(self.dl))
        else:
            return next(self.dl)

    def filt(self, args):
        idx, item = args
        fig, label = item
        temp = th.zeros_like(label)
        for s in self.select:
            select = (label == s)
            temp  += select
        select = (temp > 0)
        fig = fig[select]
        label = label[select]
        item = (fig, label)
        return idx, item


def dataGenerator(batch_size=None, root="datasets",
                  train=True, seed=None,
                  select=None, expand=False, device=None,
                  dataset='mnist', shuffle_mode=0,
                  max_batch_num=None):
    """shuffle_mode 0 : no shuffle
    1  : shuffle the first epoch
    2  : shuffle every epoch
    -1 : no shuffle, and only use the first batch
    """
    if device is None:
        device = 'cuda' if th.cuda.is_available( ) else 'cpu'
    if shuffle_mode == -1:
        data_gen = Loader(batch_size=batch_size, root=root,
                          train=train, seed=seed, select=select,
                          dataset=dataset, shuffle=(shuffle_mode > 0))
        ldata = [ ]
        for i in data_gen:
            _, data = i
            ldata.append(data)
        batch_number = 1
        while True:
            idx = 0
            source, target = ldata[0]
            if expand:
                source = expand_data(source)
            source = source.to(device)
            target = target.to(device)
            target = label2onehot(target, select=select)
            while True:
                yield idx, batch_number, source, target
    if shuffle_mode < 2:
        data_gen = Loader(batch_size=batch_size, root=root,
                          train=train, seed=seed, select=select,
                          dataset=dataset, shuffle=(shuffle_mode > 0))
        ldata = [ ]
        for idx, i in enumerate(data_gen):
            if max_batch_num is None or idx < max_batch_num:
                _, data = i
                ldata.append(data)
        batch_number = len(ldata)
        while True:
            for idx, (source, target) in enumerate(ldata):
                if expand:
                    source = expand_data(source)
                source = source.to(device)
                target = target.to(device)
                target = label2onehot(target, select=select)
                yield idx, batch_number, source, target
    elif shuffle_mode == 2:
        while True:
            data_gen = Loader(batch_size=batch_size, root=root,
                              train=train, seed=seed, select=select,
                              dataset=dataset, shuffle=True)
            ldata = [ ]
            for idx, i in enumerate(data_gen):
                if max_batch_num is None or idx < max_batch_num:
                    _, data = i
                    ldata.append(data)
            batch_number = len(ldata)
            for idx, (source, target) in enumerate(ldata):
                if expand:
                    source = expand_data(source)
                source = source.to(device)
                target = target.to(device)
                target = label2onehot(target, select=select)
                yield idx, batch_number, source, target


def example( ):
    data = Loader(batch_size=1000, train=True, select=None)
    batch_idx, (example_data, example_targets) = next(data)
    print("size of example data:", example_data.shape)
    print("size of example targets:", example_targets.shape)

    import matplotlib.pyplot as pl
    pl.figure( )
    for i in range(9):
        pl.subplot(3, 3, i+1)
        pl.tight_layout( )
        pl.imshow(example_data[i][0, :, :], cmap='gray', interpolation='none')
        pl.title("Ground Truth: {}".format(example_targets[i]))
        pl.xticks([ ])
        pl.yticks([ ])
    pl.show( )


if __name__ == '__main__':
    example( )
