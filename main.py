import argparse
from pathlib import Path
from DataLoader import dataGenerator
from ResMPS import ResMPS
from OtherNN import EM, FullLinear
from collections import namedtuple
from functools import partial
from Tools import timer
import torch as th
import dill
import sys


Ostate = namedtuple('Ostate', ('loss_train',
                               'loss_test',
                               'acc_train',
                               'acc_test',
                               ))


def cal_loss(predict, target, lossfunc, mean=True):
    batch_size, *_ = predict.shape
    if lossfunc == 'mse':
        predict2 = th.squeeze(predict)
        loss = th.sum((target-predict2)**2)
    elif lossfunc == 'nll':
        predict2 = th.squeeze(predict)
        target2  = th.argmax(target, axis=1)
        loss = th.nn.NLLLoss(reduction='sum')(predict2, target2)
    elif lossfunc == 'cross_entropy':
        predict2 = th.squeeze(predict)
        target2  = th.argmax(target, axis=1)
        loss = th.nn.CrossEntropyLoss(reduction='sum')(predict2, target2)
    else:
        raise ValueError('loss function not defined.')
    if mean:
        loss = loss/batch_size
    return loss


def cal_acc_and_loss(cf, input_data, target, lossfunc):
    with th.no_grad( ):
        predict        = cf(input_data)
        label_predict  = th.argmax(th.squeeze(predict), dim=1)
        label_target   = th.argmax(target, dim=1)
        total_number   = len(label_predict)
        true_number    = th.sum(label_predict == label_target).item( )
        loss = cal_loss(predict, target, lossfunc, mean=False).item( )
    return true_number, loss, total_number


def evaluation(cf, data_gen, lossfunc):
    total_true = 0
    total_loss = 0
    total_numb = 0
    while True:
        i, j, input_data, target = next(data_gen)
        n1, n2, n3 = cal_acc_and_loss(cf, input_data, target, lossfunc)
        total_true += n1
        total_loss += n2
        total_numb += n3
        if i == j-1:
            break
    meanloss = total_loss/total_numb
    meanacc  = total_true/total_numb
    return meanacc, meanloss


@timer
def epoch_step(cf, optim, gtrain1, gtrain2, gtest,
               step_batch_num, lstate,
               lossfunc='cross_entropy',
               scheduler=None
               ):
    # print learning rate
    lr = [i['lr'] for i in optim.param_groups]
    str_lr = ', '.join([str(i) for i in lr])
    printfunc(f"learning rate={str_lr}")

    printfunc('starting training process...')

    while True:
        step, total_step, input_data, target = next(gtrain1)
        lstr = [ ]
        lstr.append(f"(step/total)=({step+1:>4d}/{total_step:>4d}),")
        predict = cf(input_data)
        label_predict  = th.argmax(th.squeeze(predict), dim=1)
        label_target   = th.argmax(target, dim=1)
        total_number   = len(label_predict)
        true_number    = th.sum(label_predict == label_target).item( )
        lstr.append(f"batch accuracy = {true_number/total_number*100:.2f}%.")
        loss = cal_loss(predict, target, lossfunc=lossfunc)
        loss.backward( )
        printfunc('\r'+' '.join(lstr), end='', flush=True)
        if (step % step_batch_num == step_batch_num - 1) \
                or (step == total_step-1):
            with th.no_grad( ):
                cf.project_grad( )
            optim.step( )
            with th.no_grad( ):
                cf.project_tensor( )
            optim.zero_grad( )

        if step == total_step-1:
            printfunc( )
            printfunc('starting evaluating process...', flush=True)
            cf.eval( )
            acc1, loss1 = evaluation(cf, gtrain2, lossfunc)
            acc2, loss2 = evaluation(cf,   gtest, lossfunc)
            cf.train( )
            lstate.append(Ostate(loss1, loss2, acc1, acc2))
            printfunc(f'train set accuracy = {acc1*100:.2f}%')
            printfunc(f'test  set accuracy = {acc2*100:.2f}%')
            printfunc(f'train set loss     = {loss1:.5f}')
            printfunc(f'test  set loss     = {loss2:.5f}')
            if len(lstate) > 20:
                loss_list = [i[0] for i in lstate]
                loss1 = min(loss_list[-10:])
                loss2 = min(loss_list[-20:-10])
                convergence_factor = loss1/loss2
            else:
                convergence_factor = 0
            printfunc(f'convergence factor = {convergence_factor:.5f}')
            break
    return convergence_factor


def save_para(params, path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)
    th.save(params, f'{path_str}/parameter')
    with open(f'{path_str}/parameter.txt', 'w') as f:
        for key, value in params.items( ):
            f.write(f'{key}:\t{value}\n')
    return None


def run(chi=12,
        batch_size_train1=1000,
        batch_size_train2=1000,
        batch_size_test=1000,
        step_batch_num=1,
        total_step=20,
        max_prune_number=3,
        critical_prune_number=.1,
        critical_convergence_factor=.8,
        prune_mode=None,
        max_batch_num_train1=None,
        max_batch_num_train2=None,
        max_batch_num_test=None,
        lr=1e-4,
        std=1e-3,
        dataset='mnist',
        lossfunc="cross_entropy",
        network="ResMPS",
        optim_method='Adam',
        root='.',
        relu=False,
        dropout_prob=0,
        perturbation=True,
        parallel=True,
        forward_norm=1,
        rotation=False,
        norm_grad=False,
        norm_tensor=False,
        write2file=False,
        path=None,
        select=None,
        device=None,
        shuffle_mode=2,
        fm_funcs=None,
        tsne=False,
        taskname='run'
        ):

    path_str = f'{root}/result/{taskname}'
    save_para(locals( ), path_str)

    # overload print function
    original_stdout = sys.stdout
    f = open(f'{root}/result/{taskname}.log', 'w')
    global printfunc

    def printfunc(*fargs, **fkargs):
        if write2file:
            sys.stdout = f
            print(*fargs, **fkargs)
            sys.stdout = original_stdout
            print(*fargs, **fkargs)
        else:
            print(*fargs, **fkargs)

    if isinstance(select, str):
        select = [int(i) for i in select]

    if device is None:
        device = 'cuda' if th.cuda.is_available( ) else 'cpu'
    if prune_mode is not None:
        assert prune_mode in ['percentage_add', 'percentage_mul', 'magnetude']
    if select is None:
        nlabel = 10
    else:
        nlabel = len(select)
    if path == 'hilbert':
        expand = True
        n_site = 1025
    else:
        expand = False
        n_site = 785
    if fm_funcs is None or len(fm_funcs) == 0:
        d = 2
    else:
        fm_funcs = eval(fm_funcs)
        d = len(fm_funcs)

    if network in ["ResMPS"]:
        cf = ResMPS(d=d, D=chi, n_site=n_site, n_label=nlabel,
                    std=std,
                    renorm=norm_tensor,
                    renorm_grad=norm_grad,
                    rotation=rotation,
                    forward_norm=forward_norm,
                    parallel_forward=parallel,
                    perturbation=perturbation,
                    relu=relu,
                    dropout_prob=dropout_prob,
                    path=path,
                    device=device,
                    tsne=tsne,
                    feature_map_funcs=fm_funcs
                    ).to(device)
    elif network in ["LM", "FM"] + [f"ResMPS_order{i}" for i in range(1, 4)]:
        if network in ["LM", "ResMPS_order1"]:
            CF = partial(EM, order=1)
        elif network in ["FM", "ResMPS_order2"]:
            CF = partial(EM, order=2)
        elif network in ["ResMPS_order3"]:
            CF = partial(EM, order=3)
        cf = CF(D=chi, n_res=n_site-1, n_class=nlabel,
                relu=relu,
                dropout_prob=dropout_prob,
                device=device,
                ).to(device)
    elif network == "FullLinear1":
        cf = FullLinear(n_site-1, nlabel, device=device).to(device)
    elif network == "FullLinear2":
        cf = FullLinear(n_site-1, nlabel, depth=2, device=device).to(device)
    elif network == "FullLinear3":
        cf = FullLinear(n_site-1, nlabel, depth=3, device=device).to(device)
    else:
        raise KeyError(f"network {network} undefined !")
    total_params = sum(p.numel() for p in cf.parameters())
    printfunc(f"Total number of parameters is {total_params}.")

    gtrain1 = dataGenerator(root=f"{root}/datasets",
                            batch_size=batch_size_train1,
                            max_batch_num=max_batch_num_train1,
                            train=True, seed=None,
                            device=device, expand=expand,
                            select=select,
                            dataset=dataset,
                            shuffle_mode=shuffle_mode
                            )
    gtrain2 = dataGenerator(root=f"{root}/datasets",
                            batch_size=batch_size_train2,
                            max_batch_num=max_batch_num_train2,
                            train=True, seed=None,
                            device=device, expand=expand,
                            select=select,
                            dataset=dataset,
                            shuffle_mode=shuffle_mode
                            )
    gtest = dataGenerator(root=f"{root}/datasets",
                          batch_size=batch_size_test,
                          max_batch_num=max_batch_num_test,
                          train=False, seed=None,
                          device=device, expand=expand,
                          select=select,
                          dataset=dataset,
                          shuffle_mode=shuffle_mode
                          )

    optim_func_dict = {
        "SGD": th.optim.SGD,
        "Adam": th.optim.Adam
    }
    optim = optim_func_dict[optim_method](cf.parameters( ), lr=lr)

    lstate = [ ]

    prune_step = 0
    prune_count = 0
    for epoch in range(1, total_step+1):
        prune_step += 1
        printfunc('\nstep', epoch, flush=True)
        convergence_factor = epoch_step(cf, optim, gtrain1, gtrain2, gtest,
                                        step_batch_num, lstate,
                                        lossfunc=lossfunc
                                        )
        th.save({
            'epoch': epoch,
            'model_state_dict': cf.state_dict( ),
            'optimizer_state_dict': optim.state_dict( ),
            'lstate': lstate
        }, f'{path_str}/checkpoint.tar')
        if tsne:
            cf_copy=dill.dumps(cf)
            th.save(cf_copy, f'{path_str}/network.tar')
            # th.save(cf, f'{path_str}/network.tar', pickle_protocol=)
        if convergence_factor > critical_convergence_factor \
                and prune_step > 25 \
                and prune_mode is not None:
            th.save({
                'epoch': epoch,
                'model_state_dict': cf.state_dict( ),
                'optimizer_state_dict': optim.state_dict( ),
                'lstate': lstate
            }, f'{path_str}/checkpoint_prune{prune_count}.tar')
            if prune_count > max_prune_number:
                break
            if prune_mode == 'percentage_add':
                temp = critical_prune_number*(prune_count + 1)
                cf.prune(temp, mode='percent')
            elif prune_mode == 'percentage_mul':
                temp = 1 - (1 - critical_prune_number)**(prune_count + 1)
                cf.prune(temp, mode='percent')
            elif prune_mode == 'magnetude':
                cf.prune(critical_prune_number/(1+prune_count),
                         mode='magnetude')
            else:
                raise ValueError(f"prune mode {prune_mode} undefined!")
            prune_step = 0
            prune_count += 1

    if write2file:
        f.close( )


if __name__ == "__main__":
    parser = argparse.ArgumentParser( )
    # dimension of hidden feature chi
    parser.add_argument('--chi', type=int, default=12,
                        help='dimension of hidden feature chi')
    # total training steps
    parser.add_argument('--total_step', type=int, default=80,
                        help='total training steps')
    # batch size
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch size')
    # learning rate
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    # standard deviation of initial weights
    parser.add_argument('--std', type=float, default=1e-3,
                        help='standard deviation of initial weights')
    # mnist or fashion_mnist
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        help='mnist or fashion_mnist')
    # ResMPS, ResMPS_order1, ResMPS_order2 or ResMPS_order3
    parser.add_argument('--network', type=str, default='ResMPS',
                        help='ResMPS, ResMPS_order1, ResMPS_order2 or ResMPS_order3')
    # Name your task as you like
    parser.add_argument('--taskname', type=str, default='example_task',
                        help='Name your task as you like')
    # normal, zigzag or random
    parser.add_argument('--path', type=str, default='normal',
                        help='normal, zigzag or random')
    # whether to use parallel acceleration, only valid for linear models
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='whether to use parallel acceleration, only valid for linear models')
    # whether to use the traditional MPS model, see https://arxiv.org/abs/1906.06329
    parser.add_argument('--classical', default=False, action='store_true',
                        help='whether to use the traditional MPS model, see https://arxiv.org/abs/1906.06329')
    # whether to use ReLU, True or False
    parser.add_argument('--relu', default=False, action='store_true',
                        help='whether to use ReLU, True or False')
    # a float between 0 and 1, 0 is no dropout
    parser.add_argument('--dropout_prob', type=float, default=0,
                        help='a float between 0 and 1, 0 is no dropout')
    # cuda or cpu
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda or cpu')
    # string, which will be conveted to a tuple of lambda functions
    parser.add_argument('--feature_map', type=str, default='(lambda x: x,)',
                        help='string, which will be conveted to a tuple of lambda functions')
    # if true, collect intermediate hidden variables
    parser.add_argument('--tsne', default=False, action='store_true',
                        help='if true, collect intermediate hidden variables')
    # if true, write output to a log file
    parser.add_argument('--write2file', default=False, action='store_true',
                        help='if true, write output to a log file')
    args = parser.parse_args( )

    run(chi=args.chi,
        total_step=args.total_step,
        batch_size_train1=args.batch_size,
        batch_size_train2=args.batch_size,
        batch_size_test=args.batch_size,
        lr=args.lr,
        std=args.std,
        parallel=args.parallel,
        dataset=args.dataset,
        network=args.network,
        relu=args.relu,
        dropout_prob=args.dropout_prob,
        device=args.device,
        write2file=args.write2file,
        path=args.path,
        fm_funcs=args.feature_map,
        perturbation=(not args.classical),
        tsne=args.tsne,
        taskname=args.taskname
        )
