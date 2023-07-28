import configparser
import sys
import os
import shutil

usage = "Usage: main.py [train|test] <config file>"
assert len(sys.argv) in [2, 3], usage

mode = sys.argv[1]

args = configparser.ConfigParser()
args.read('default.ini')

if len(sys.argv) == 3:
    conf_file = sys.argv[2]
    args.read(conf_file)
    assert os.path.isfile(conf_file), "No such config file"

output_base_dir = args['checkpoints']['global_output_folder']
output_dir = os.path.join(output_base_dir, args['checkpoints']['base_name'])
args['checkpoints']['output_dir'] = output_dir

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'conf.ini'), 'w') as f:
    args.write(f)

if mode == 'train':
    import train
    train.train(args)
elif mode == 'test':
    import evaluation
    evaluation.test(args)

elif mode.startswith('wim'):
    wim_output_dir = os.path.join(output_dir, mode)
    args['checkpoints']['wim_output_dir'] = wim_output_dir
    os.makedirs(wim_output_dir, exist_ok=True)

    wim_ood = mode.split('-')[1:]
    import evaluation

    if not os.path.exists(os.path.join(wim_output_dir, 'model.pt')):
        print('>> Wim backprop to be done with', *wim_ood)
        evaluation.wim.wim_train(args, *wim_ood)
    else:
        print('>> Wim backprop already done')

    args['checkpoints']['output_dir'] = wim_output_dir
    evaluation.test(args)

elif mode == 'generate':
    import evaluation.generation
    evaluation.generation.main(args)
else:
    print(usage)
    sys.exit(1)
