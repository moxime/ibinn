import json
from os.path import join
import numpy as np


def to_latex_table_row(results_dict, out_dir, name="",
                       italic_ood=False,
                       blank_ood=False,
                       italic_entrop=False,
                       blank_classif=False,
                       blank_bitspdim=False):

    name = name.replace("_", " ")
    outfile = open(join(out_dir, 'results.tex'), 'w')

    outfile.write(" & {:>14s} &\n".format(name))
    if blank_classif:
        acc_str = '--'
    else:
        acc_str = '{:.2f}'.format(100. - results_dict['test_metrics']['accuracy'])

    if blank_bitspdim:
        bits_str = '--'
    else:
        bits_str = '{:.2f}'.format(results_dict['test_metrics']['bits_per_dim'])

    outfile.write("{:>14s}  {:>14s} &{:>14s} &\n".format(' ', acc_str, bits_str))

    ce = results_dict['calib_err']
    if blank_classif:
        outfile.write(("{:>14s} &" * 4 + "\n").format('--', '--', '--', '--'))
    else:
        outfile.write(("{:>14.2f} &" * 4 + "\n").format(ce['gme'], ce['ece'], ce['mce'], ce['ice']))

    ood = results_dict['ood_tt']
    ent = results_dict['ood_d_ent']

    if italic_entrop:
        se_m = '( \\it {:.2f}'.format(ent['ari_mean'])
        se_rot = '\\it {:.2f}'.format(ent['rot_rgb'])
        se_qui = '\\it {:.2f}'.format(ent['quickdraw'])
        se_noi = '\\it {:.2f}'.format(ent['noisy'])
        se_img = '\\it {:.2f})'.format(ent['imagenet'])
        outfile.write(("{:>14s} &" * 5 + "\n").format(se_m, se_rot, se_qui, se_noi, se_img))
    else:
        outfile.write(("{:>14.2f} &" * 5 + "\n").format(ent['ari_mean'],
                                                        ent['rot_rgb'],
                                                        ent['quickdraw'],
                                                        ent['noisy'],
                                                        ent['imagenet']))

    if italic_ood or blank_ood:
        if italic_ood:
            oo_m = '( \\it {:.2f}'.format(100. * ood['ari_mean'])
            oo_rot = '\\it {:.2f}'.format(100. * ood['rot_rgb'])
            oo_qui = '\\it {:.2f}'.format(100. * ood['quickdraw'])
            oo_noi = '\\it {:.2f}'.format(100. * ood['noisy'])
            oo_img = '\\it {:.2f})'.format(100. * ood['imagenet'])
        else:
            oo_m, oo_rot, oo_qui, oo_noi, oo_img = ['--'] * 5

        outfile.write((("{:>14s} &" * 5)[:-1] + "\\\\\n").format(oo_m, oo_rot, oo_qui, oo_noi, oo_img))

    else:
        outfile.write((("{:>14.2f} &" * 5)[:-1] + "\\\\\n").format(100. * ood['geo_mean'],
                                                                   100. * ood['rot_rgb'],
                                                                   100. * ood['quickdraw'],
                                                                   100. * ood['noisy'],
                                                                   100. * ood['imagenet']))

    outfile.close()


def to_csv(results_dict, out_dir):
    pass


def to_df(results_dict):
    import pandas as pd

    df = pd.concat({k: pd.DataFrame(results_dict[k]).T for k in results_dict}, axis=1)
    # df_ = pd.DataFrame(df.values.tolist())
    df.columns.rename(['method', 'measures'], inplace=True)
    df.index.rename('set', inplace=True)
    df = df.unstack('set').to_frame().T.reorder_levels(['set', 'method', 'measures'], axis=1)
    return df.reindex(columns=sorted(df.columns))

    df = pd.concat({k: pd.DataFrame(results_dict[k]).T for k in results_dict if 'ood' not in k}, axis=1)
    return df


def to_csv_row(results_dict, out_dir):
    pass


def to_raw(results_dict, out_dir):
    pass


def to_json(results_dict, out_dir):
    json.dump(results_dict, open(join(out_dir, 'results.json'), 'w'),
              sort_keys=True, indent=2)


def to_console(results_dict, out_dir):

    testset = next(iter(results_dict['calib_err']))
    logfile = open(join(out_dir, 'results.log'), 'w')
    ce = results_dict['calib_err'][testset]
    ece, mce, ice, ovc, gme = ce['ece'], ce['mce'], ce['ice'], ce['oce'], ce['gme']

    def log_write(line, endline='\n'):
        print("\t" + line, flush=True)
        logfile.write(line)
        logfile.write(endline)

    log_write('ACC     %.4f' % (results_dict['test_metrics'][testset]['accuracy']))
    log_write('BITS    %.4f' % (results_dict['test_metrics'][testset]['bits_per_dim']))
    log_write('')

    log_write(('XCE     ' + '%-10s' * 4) % ('ECE', 'MCE', 'ICE', 'OVC'))
    log_write(('XCE     ' + '%-10.6f' * 4) % (ece, mce, ice, ovc))
    log_write('XCE GM  %.6f' % (21.5443 * gme))
    log_write('')

    for i, test_type in enumerate(['ood_ent', 'ood_d_ent', 'ood_1t', 'ood_2t', 'ood_tt']):
        rocs = results_dict[test_type]
        labels_list = list(rocs.keys())

        if i == 0:
            log_write('DATASET    ' + ''.join(['%-16s' % (l) for l in labels_list]))

        if '_ent' in test_type:
            mult = 1.
        else:
            mult = 100.

        log_write('{:9s} '.format(test_type.upper()) +
                  ''.join(['{:-16.4f}'.format(mult * rocs[l]['auc' if 'auc' in rocs[l] else 'val']) for l in labels_list]))

    logfile.close()
