from model import GenerativeClassifier
import matplotlib.pyplot as plt
import os

import torch
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')


def outlier_detection(inn_model, data, args, test_set=False, target_tpr=0.95):
    ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

    oodsets = args['evaluation']['oodsets'].split()

    # import ood_datasets.imagenet
    # import ood_datasets.cifar
    # import ood_datasets.quickdraw
    import ood_datasets.svhn
    import ood_datasets.lsunr
    import ood_datasets.lsunc
    import ood_datasets.cifar100
    import ood_datasets.const_noise

    ensemble = int(args['evaluation']['ensemble_members'])
    inn_ensemble = [inn_model]

    if ensemble > 1:
        print('>> Loading WAIC ensemble', end='')
        for i in range(1, ensemble):
            print('.', end='', flush=True)
            inn_ensemble.append(GenerativeClassifier(inn_model.args))
            model_fname = os.path.join(args['checkpoints']['output_dir'], 'model.%.2i.pt' % (i))
            inn_ensemble[-1].load(model_fname)
            inn_ensemble[-1].cuda()
            inn_ensemble[-1].eval()

    def collect_scores(generator, oversample=1):
        with torch.no_grad():
            scores_cumul = []
            entrop_cumul = []
            for n in range(oversample):
                for x in generator:
                    x = x[0].cuda()
                    ll_joint = []
                    for inn in inn_ensemble:
                        losses = inn(x, y=None, loss_mean=False)
                        ll_joint.append(losses['L_x_tr'].cpu().numpy())
                    entrop = torch.sum(- torch.softmax(losses['logits_tr'], dim=1)
                                       * torch.log_softmax(losses['logits_tr'], dim=1), dim=1).cpu().numpy()

                    ll_joint = np.stack(ll_joint, axis=1)
                    scores_cumul.append(np.mean(ll_joint, axis=1) + np.var(ll_joint, axis=1))
                    entrop_cumul.append(entrop)
        return np.concatenate(scores_cumul), np.concatenate(entrop_cumul)

    quantile_resolution = 200
    quantile_steps = np.concatenate([np.linspace(0.0, 0.1, num=quantile_resolution, endpoint=False),
                                     np.linspace(0.1, 0.9, num=quantile_resolution, endpoint=False),
                                     np.linspace(0.9, 1.0, num=quantile_resolution + 1, endpoint=True)])

    scores_all = {}
    entrop_all = {}
    generators = []

    in_distrib_data = (data.test_loader if test_set else [(data.val_x, torch.argmax(data.val_y, dim=1))])

    # generators = [
    #                   #(data.test_loader, 'test'),
    #                   (ood_datasets.cifar.cifar_rgb_rotation(inn_model.args, 0.35), 'rot_rgb'),
    #                   (ood_datasets.quickdraw.quickdraw_colored(inn_model.args), 'quickdraw'),
    #                   (ood_datasets.cifar.cifar_noise(inn_model.args, 0.015), 'noisy'),
    #                   (ood_datasets.imagenet.imagenet(inn_model.args), 'imagenet'),
    #                   #(ood_datasets.svhn.svhn(inn_model.args), 'SVHN'),
    #                   ]

    if 'svhn' in oodsets:
        generators.append((ood_datasets.svhn.svhn(inn_model.args), 'svhn'))

    if 'lsunr' in oodsets:
        generators.append((ood_datasets.lsunr.lsunr(inn_model.args), 'lsunr'))

    if 'lsunc' in oodsets:
        generators.append((ood_datasets.lsunc.lsunc(inn_model.args), 'lsunc'))

    if 'cifar100' in oodsets:
        generators.append((ood_datasets.cifar100.cifar100(inn_model.args), 'cifar100'))

    if 'const' in oodsets:
        generators.append((ood_datasets.const_noise.const32(inn_model.args), 'const32'))

    if 'noise' in oodsets:
        generators.append((ood_datasets.const_noise.uniform32(inn_model.args), 'uniform32'))

    for gen, label in generators:
        print(f'>> Computing OoD score for {label}')
        scores_all[label], ent = collect_scores(gen)
        entrop_all[label] = {'val': np.mean(ent)}

    scores_ID, entrop_ID = collect_scores(in_distrib_data, oversample=int(args['evaluation']['train_set_oversampling']))
    entrop_ID = np.mean(entrop_ID)
    quantiles_ID = np.quantile(scores_ID, quantile_steps)
    typical_ID = np.mean(scores_ID)
    typicality_scores_ID = np.abs(scores_ID - typical_ID)

    fig_roc = plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='random')
    plt.grid(True)

    fig_hist = plt.figure(figsize=(8, 6))
    plt.hist(scores_ID, bins=50, histtype='step', density=True, color='gray', label='orig. distrib.')

    def auc(test_scores, train_scores, label=''):
        xjoint = np.sort(np.concatenate((train_scores, test_scores)))
        xjoint[0] -= 0.0001
        val_range = (np.min(xjoint), np.max(xjoint))

        roc = []
        target_fpr = None
        for x in xjoint:
            tpr = np.mean(train_scores < x)
            fpr = np.mean(test_scores < x)
            if tpr >= target_tpr and target_fpr is None:
                target_fpr = fpr
                print('{:.1%}: {:.1%}'.format(tpr, target_fpr))
            roc.append((fpr, tpr))
        roc = np.array(roc).T

        auc = np.trapz(roc[1], x=roc[0])
        plt.figure(fig_hist.number)
        plt.hist(test_scores, bins=50, histtype='step', density=True, label=label + ' (%.4f AUC)' % (auc))
        plt.figure(fig_roc.number)
        plt.plot(roc[0], roc[1], label=label + ' (%.4f AUC)' % (auc))

        return {'auc': 100 * auc, 'fpr': 100 * target_fpr}

    def auc_quantiles(test_scores, train_quantiles, quantile_steps):

        roc = []
        target_fpr = None

        for i in range(len(quantile_steps) // 2):
            i_ = len(quantile_steps) // 2 - i
            tpr = (quantile_steps[-i_] - quantile_steps[i_ - 1])
            fpr = np.mean(np.logical_and(test_scores >= train_quantiles[i_ - 1], test_scores <= train_quantiles[-i_]))
            if tpr >= target_tpr and target_fpr is None:
                target_fpr = fpr
                print('{:.1%}: {:.1%}'.format(tpr, fpr))

            roc.append((fpr, tpr))
        roc.append((1., 1.))

        roc = np.array(roc).T
        auc = np.trapz(roc[1], x=roc[0])
        return {'auc': 100 * auc, 'fpr': 100 * target_fpr}

    aucs_one_tailed = {}
    aucs_two_tailed = {}
    aucs_typicality = {}
    delta_entropy = {}
    for label, score in scores_all.items():
        print('{} 1T'.format(label))
        aucs_one_tailed[label] = auc(score, scores_ID, label)
        print('{} 2T'.format(label))
        aucs_two_tailed[label] = auc_quantiles(score, quantiles_ID, quantile_steps)
        print('{} TT'.format(label))
        aucs_typicality[label] = auc(np.abs(score - typical_ID), typicality_scores_ID)
        delta_entropy[label] = {'val': entrop_all[label]['val'] - entrop_ID}

    plt.figure(fig_hist.number)
    plt.legend()
    plt.figure(fig_roc.number)
    plt.legend()

    return aucs_one_tailed, aucs_two_tailed, aucs_typicality, entrop_all, delta_entropy
