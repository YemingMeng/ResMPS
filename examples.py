from main import run
from itertools import product
from tsne import tsne_plot

# Example code to calculate the result of paper arXiv:2012.11841

# Figure 2, Figure 3
def example1():
    run(chi=40, dataset="fashion_mnist", total_step=50, lr=1e-4, parallel=False,
        network="ResMPS", optim_method="Adam", write2file=True,
        fm_funcs="(lambda x:x, )",
        tsne=True,
        taskname="tsne")
    tsne_plot(taskname="tsne")

# Figure 4(a), Table 1, Table 2
def example2():
    ldata = ["mnist", "fashion_mnist"]
    lfeaturemap = ["(lambda x:x, )", "(lambda x:x, lambda x:1-x)"]
    lrelu = [True, False]
    ldropout_prob = [0, 0.6]
    for idx, (data, featuremap, relu, dropout) \
        in enumerate(product(ldata, lfeaturemap, lrelu, ldropout_prob)):
        nchannel = len(eval(featuremap))
        run(chi=100, dataset=data, total_step=200, lr=2e-4/nchannel, parallel=False,
            network="ResMPS", optim_method="Adam", write2file=True,
            fm_funcs=featuremap,
            relu=relu,
            dropout_prob=dropout,
            taskname=f"run{idx+1}")

# Figure 4(b)
def example3():
    run(chi=40, dataset="fashion_mnist", total_step=1e100, lr=1e-4, parallel=False,
        network="ResMPS", optim_method="Adam", write2file=True,
        fm_funcs="(lambda x:x, )", taskname="pruning",
        max_prune_number=30,
        critical_prune_number=0.2,
        critical_convergence_factor=1,
        prune_mode="percentage_mul",
        )

# Figure 5
def example4():
    lepsilon = [0.02, 0.04, 0.06, 0.08, 0.10]
    for epsilon in lepsilon:
        run(chi=40, dataset="fashion_mnist", total_step=50, lr=1e-4, parallel=False,
            network="ResMPS", optim_method="Adam", write2file=True,
            fm_funcs="(lambda x:x, )",
            taskname=f"epsilon{epsilon}")

# Figure 6
def example5():
    lnetwork = ["ResMPS_order1", "ResMPS_order2", "ResMPS_order3", "ResMPS"]
    lchi = [10, 20, 30, 40]
    for chi, network in product(lchi, lnetwork):
        run(chi=100, dataset="fashion_mnist", total_step=200, lr=1e-4, parallel=False,
            network="ResMPS", optim_method="Adam", write2file=True,
            fm_funcs="(lambda x:x, )",
            taskname=f"{network}_chi{chi}")


# Figure 8
def example6():
    lpath = ["zigzag", "hilbert", "random"]
    for path in lpath:
        run(chi=40, dataset="fashion_mnist", total_step=200, lr=1e-4, parallel=False,
            network="ResMPS", optim_method="Adam", write2file=True,
            fm_funcs="(lambda x:x, )",
            path=path,
            taskname=f"path_{path}",
            )


if __name__ == "__main__":
    # Uncomment to run the corresponding examples
    example1( )
    # example2( )
    # example3( )
    # example4( )
    # example5( )
    # example6( )
