from . import svhn, lsunr, cifar100
import data


def get_ood_datasets(args, *oodsets, include_testset=True):

    generators = []

    if include_testset:

        generators.append(data.Dataset(args))

    for s in oodsets:
        if s == 'svhn':
            generators.append((svhn(args), 'svhn'))

    if s == 'lsunr':
        generators.append((lsunr(args), 'lsunr'))

    return generators
