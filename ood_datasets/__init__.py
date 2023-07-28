from . import svhn, lsunr
import data


def get_ood_datasets(args, *oodsets, include_testset=True):

    generators = []

    if include_testset:

        generators.append(data.Dataset(args))

    for s in oodsets:
        if s == 'svhn':
            generators.append((svhn(args), 'SVHN'))

    if s == 'lsunr':
        generators.append((lsunr(args), 'LSUNR'))

    return generators
