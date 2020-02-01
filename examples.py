import pandas as pd
import numpy as np

import stad
import stad.visualize
from stad.util import normalise_number_between_0_and_255, hex_to_hsv, calculate_highD_dist_matrix

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
        print("Dataset not known")


def main():
    values, lens, features = load_testdata('circles')
    highD_dist_matrix = calculate_highD_dist_matrix(values)
    g = stad.run_stad(highD_dist_matrix, lens=lens, features=features)
    stad.visualize.draw_stad(g)


if __name__ == '__main__':
    main()