import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calculate_highD_dist_matrix(data):
    non_normalized = euclidean_distances(data)
    max_value = np.max(non_normalized)
    return non_normalized/max_value

def rgb_to_hsv(rgb):
    r = float(rgb[0])
    g = float(rgb[1])
    b = float(rgb[2])
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return list(int(hex[i:i+int(hlen/3)], 16) for i in range(0, hlen, int(hlen/3)))

def hex_to_hsv(hex):
    return rgb_to_hsv(hex_to_rgb(hex))

def normalise_number_between_0_and_1(nr, domain_min, domain_max):
    return (nr-domain_min)/(domain_max - domain_min)

def normalise_number_between_0_and_255(nr, domain_min, domain_max):
    return 255*normalise_number_between_0_and_1(nr, domain_min, domain_max)