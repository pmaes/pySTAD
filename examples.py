import stad
import pandas as pd

def main():
    data = pd.read_csv('data/five_circles.csv', header=0)
    values = data[['x','y']].values.tolist()
    lens = data['hue'].map(lambda x:stad.hex_to_hsv(x)[0]).values
    xs = data['x'].values.tolist()
    ys = data['y'].values.tolist()
    hues = data['hue'].values.tolist()
    highD_dist_matrix = stad.calculate_highD_dist_matrix(values)
    g = stad.run_stad(highD_dist_matrix, lens=lens, features={'x':xs, 'y':ys, 'hue': hues})
    # g = run_stad(highD_dist_matrix, features={'x':xs, 'y':ys, 'hue':hues})
    stad.draw_stad(g)

if __name__ == '__main__':
    main()