import torch as th
from sklearn import manifold
import dill
from DataLoader import dataGenerator
import matplotlib.pyplot as pl
from matplotlib import colors


def tsne_plot(taskname="tsne"):
    method = manifold.TSNE(n_components=2)
    # load resmps model from file
    cf_copy = th.load(f'result/{taskname}/network.tar')
    cf=dill.loads(cf_copy)
    # prepare data loader
    datagen = dataGenerator(batch_size=1000,
                            device=cf.device,
                            dataset="fashion_mnist"
                            )
    _, _, input_data, target = next(datagen)
    with th.no_grad( ):
        output_data = cf(input_data)

    hiddens = [i.to('cpu') for i in cf.hiddens]
    label = th.argmax(target, dim=1).to('cpu')

    # t-SNE dimensional reduction
    hidden = hiddens[-1]
    hidden = hidden.detach( ).numpy( )
    a, *_ = hidden.shape
    tsne_result = method.fit_transform(hidden.reshape(a, -1))
    x_min, x_max = tsne_result.min(0), tsne_result.max(0)
    tsne_result = (tsne_result - x_min) / (x_max - x_min) - 0.5

    color    = [pl.cm.Set1(i) for i in range(10)]
    color[9] = colors.to_rgba('darkviolet')
    for i in range(len(tsne_result)):
        x, y = tsne_result[i, 0], tsne_result[i, 1]
        x += 0.5
        y += 0.5
        pl.text(x, y, str(label[i].item()),
                color=color[label[i]],
                fontdict={'weight': 'bold', 'size': 9})
    pl.xticks([])
    pl.yticks([])
    ax = pl.gca( )
    ax.axis('off')
    pl.show( )
