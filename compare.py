import time

import click
import pandas as pd
import numpy as np

import stad
import stad.optimized
from stad.util import normalise_number_between_0_and_255, hex_to_hsv, calculate_highD_dist_matrix

# This script encapsulates the loading of test datasets, switching between STAD
# implementations, and visualising the resulting graph.
# It offers a small command line interface. Run `python compare.py --help` for more information.

# To add a dataset, add a branch to load_testdata and add it to the dataset
# argument choices on main().

def load_testdata(dataset):
    if dataset == 'horse':
        data = pd.read_csv('data/horse.csv', header=0)
        data = data.sample(n=1000)
        values = data[['x','y','z']].values.tolist()
        x_min = min(data['x'])
        x_max = max(data['x'])
        # zs = data['z'].values
        lens = data['x'].map(lambda x: normalise_number_between_0_and_255(x, x_min, x_max)).values
        return (values, lens, {})
    elif dataset == 'simulated':
        data = pd.read_csv('data/sim.csv', header=0)
        values = data[['x','y']]
        lens = np.zeros(1)
        return (values, lens, {})
    elif dataset == 'circles':
        data = pd.read_csv('data/five_circles.csv', header=0)
        values = data[['x','y']].values.tolist()
        lens = data['hue'].map(lambda x: hex_to_hsv(x)[0]).values
        features={
            'x': data['x'].values.tolist(),
            'y': data['y'].values.tolist(),
            'hue': data['hue'].values.tolist()
        }
        return (values, lens, features)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

# To implement implementation switching, we can use python's first-order
# modules. Since these can be placed in a dict like any other object, we simply
# map from string to implementation. These both implement the same run_stad API
# function.

IMPLEMENTATIONS = {
    'base': stad,
    'optimized': stad.optimized
}


@click.command()
@click.argument('implementation', type=click.Choice(IMPLEMENTATIONS.keys(), case_sensitive=False), default='base')
# This is where new datasets should be added as an option.
@click.argument('dataset', type=click.Choice(['circles', 'horse', 'simulated'], case_sensitive=False))
@click.option('--debug/--no-debug', default=True)
@click.option('--viz/--no-viz', default=True)
@click.option('--lens/--no-lens', 'use_lens', default=False)
@click.option('--algo', type=click.Choice(['lipo', 'diffcurve'], case_sensitive=False), default='diffcurve')
def main(dataset, implementation, debug, viz, use_lens, algo):
    values, lens, features = load_testdata(dataset)

    # Ignore the loaded lens if asked. Could be optimized to only load lens if
    # asked instead of ignoring it later.
    if not use_lens:
        lens = []

    highD_dist_matrix = calculate_highD_dist_matrix(values)

    # This is where we pick which implementation to use.
    # impl_module will point to the right module, which we can then call run_stad on.
    impl_module = IMPLEMENTATIONS[implementation]
    if debug: print(f"Using '{implementation}' implementation in {impl_module}")

    opts = {
        'optimizer': algo
    }

    t_start = time.time()
    g = impl_module.run_stad(highD_dist_matrix, lens=lens, features=features, debug=debug, opts=opts)
    t_done = time.time()
    t_delta = t_done - t_start
    print(f"STAD calculation took {t_delta:.2f}s for {highD_dist_matrix.shape[0]} datapoints")

    if viz:
        # Only import the visualize module (and vega dependency) if necessary,
        # as it can take a couple seconds.
        import stad.visualize
        stad.visualize.draw_stad(g)


if __name__ == '__main__':
    main()