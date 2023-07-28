from os.path import join
from time import time
import sys

import numpy as np
from tqdm import tqdm
import torch.optim

import data
from model import GenerativeClassifier
from VIB import WrapperVIB
import evaluation


def wim_train(args):

    N_epochs = eval(args['wim']['n_epochs'])
    beta = eval(args['training']['beta_IB'])
    train_nll = bool(not eval(args['ablations']['no_NLL_term']))
    train_class_nll = eval(args['ablations']['class_NLL'])
    label_smoothing = eval(args['data']['label_smoothing'])
    grad_clip = eval(args['training']['clip_grad_norm'])

    interval_log = eval(args['checkpoints']['interval_log'])
    interval_checkpoint = eval(args['checkpoints']['interval_checkpoint'])
    interval_figure = eval(args['checkpoints']['interval_figure'])
    save_on_crash = eval(args['checkpoints']['checkpoint_when_crash'])

    output_dir = args['checkpoints']['wim_output_dir']
    resume_dir = args['checkpoints']['output_dir']

    logfile = open(join(output_dir, 'losses.dat'), 'w')
    live_loss = eval(args['checkpoints']['live_updates'])

    # args['training']['train_mu'] = 'False'

    inn = GenerativeClassifier(args)

    # inn.mu.requires_grad_(False)

    inn.cuda()
    dataset = data.Dataset(args)

    inn.load(join(resume_dir, 'model.pt'))

    def log_write(line, endline='\n'):
        print(line, flush=True)
        logfile.write(line)
        logfile.write(endline)

    plot_columns = ['time', 'epoch', 'iteration',
                    'L_x_tr',
                    'L_x_val',
                    'L_y_tr',
                    'L_y_val',
                    'acc_tr',
                    'acc_val',
                    'delta_mu_val']

    train_loss_names = [l for l in plot_columns if l[-3:] == '_tr']
    val_loss_names = [l for l in plot_columns if l[-4:] == '_val']

    header_fmt = '{:>15}' * len(plot_columns)
    output_fmt = '{:15.1f}      {:04d}/{:04d}      {:04d}/{:04d}' + '{:15.5f}' * (len(plot_columns) - 3)
    output_fmt_live = '{:15.1f}      {:04d}/{:04d}      {:04d}/{:04d}'
    for l_name in plot_columns[3:]:
        if l_name in train_loss_names:
            output_fmt_live += '{:15.5f}'
        else:
            output_fmt_live += '{:>15}'.format('')

    if eval(args['training']['exponential_scheduler']):
        print('Using exponential scheduler')
        sched = torch.optim.lr_scheduler.StepLR(inn.optimizer, gamma=0.002 ** (1 / N_epochs), step_size=1)
    else:
        print('Using milestone scheduler')
        sched = torch.optim.lr_scheduler.MultiStepLR(inn.optimizer, gamma=0.1,
                                                     milestones=eval(args['training']['scheduler_milestones']))

    for _ in range(eval(args['training']['n_epochs'])):
        sched.step()

    print('LR updatted to', *sched._last_lr)

    log_write(header_fmt.format(*plot_columns))

    t_start = time()
    if train_nll:
        beta_x = 2. / (1 + beta)
        beta_y = 2. * beta / (1 + beta)
    else:
        beta_x, beta_y = 0., 1.

    try:
        for i_epoch in range(N_epochs):

            running_avg = {l: [] for l in train_loss_names}

            mu_copy = inn.mu.clone()

            for i_batch, (x, l) in enumerate(dataset.train_loader):

                x, y = x.cuda(), dataset.onehot(l.cuda(), label_smoothing)

                with torch.no_grad():
                    losses = inn(x, y)

                if train_class_nll:
                    loss = 2. * losses['L_cNLL_tr']
                else:
                    loss = beta_x * losses['L_x_tr'] - beta_y * losses['L_y_tr']
                # loss.backward()

                torch.nn.utils.clip_grad_norm_(inn.trainable_params, grad_clip)
                # inn.optimizer.step()
                # inn.optimizer.zero_grad()

                if live_loss:
                    print(output_fmt_live.format(*([(time() - t_start) / 60.,
                                                    i_epoch, N_epochs,
                                                    i_batch, len(dataset.train_loader)]
                                                   + [losses[l].item() for l in train_loss_names])),
                          flush=True, end='\r')

                for l_name in train_loss_names:
                    running_avg[l_name].append(losses[l_name].item())

                if not i_batch % interval_log:
                    for l_name in train_loss_names:
                        running_avg[l_name] = np.mean(running_avg[l_name])

                    val_losses = inn.validate(dataset.val_x, dataset.val_y)
                    for l_name in val_loss_names:
                        running_avg[l_name] = val_losses[l_name].item()

                    losses_display = [(time() - t_start) / 60.,
                                      i_epoch, N_epochs,
                                      i_batch, len(dataset.train_loader)]

                    losses_display += [running_avg[l] for l in plot_columns[3:]]
                    # TODO visdom?
                    log_write(output_fmt.format(*losses_display))
                    running_avg = {l: [] for l in train_loss_names}
                break
            sched.step()
            dmu = (inn.mu - mu_copy).norm() / mu_copy.norm()
            print('[D] >> dmu={:.1e}'.format(dmu))

            if i_epoch > 2 and (val_losses['L_x_val'].item() > 1e5 or not np.isfinite(val_losses['L_x_val'].item())):
                if high_loss:
                    raise RuntimeError("loss is astronomical")
                else:
                    high_loss = True
            else:
                high_loss = False

            if i_epoch > 0 and (i_epoch % interval_checkpoint) == 0:
                inn.save(join(output_dir, f'model_{i_epoch}.pt'))
            if (i_epoch % interval_figure) == 0:
                evaluation.val_plots(join(output_dir, f'figs_{i_epoch}.pdf'), inn, dataset)
    except:
        if save_on_crash:
            inn.save(join(output_dir, f'model_ABORT.pt'))
        raise
    finally:
        logfile.close()

    try:
        for k in list(inn.inn._buffers.keys()):
            if 'tmp_var' in k:
                # print('>> Deleting', k)
                del inn.inn._buffers[k]
    except AttributeError:
        # Feed-forward nets dont have the wierd FrEIA problems, skip
        pass

    inn.save(join(output_dir, f'model.pt'))
