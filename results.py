import os
import pandas as pd
import json


def to_df(results_dict):
    import pandas as pd

    df = pd.concat({k: pd.DataFrame(results_dict[k]).T for k in results_dict}, axis=1)
    # df_ = pd.DataFrame(df.values.tolist())
    df.columns.rename(['method', 'measures'], inplace=True)
    df.index.rename('set', inplace=True)
    df = df.unstack('set').to_frame().T.reorder_levels(['set', 'method', 'measures'], axis=1)
    df.index = ['ibinn']
    df.index.rename('type', inplace=True)
    return df.reindex(columns=sorted(df.columns))


def to_csv(results_dict, out_dir):
    df = to_df(results_dict)
    df.to_csv(os.path.join(out_dir, 'results.csv'))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('res_dir')
    args = parser.parse_args()

    json_file = os.path.join(args.res_dir, 'results.json')

    with open(json_file) as f:
        d = json.load(f)

    to_csv(d, args.res_dir)
