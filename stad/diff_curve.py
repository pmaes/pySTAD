import numpy as np

def normalize(a):
        return (a - np.min(a)) / (np.max(a) - np.min(a))


def trace_diff_curve(f, a, b, steps, frac):
    xs, ys = [], []
    
    for x in np.linspace(a, b, steps, dtype=np.int):
        y = f(x)
        xs.append(x)
        ys.append(y)
        
        if x == 0:
            continue
        
        ay = np.array(ys)
        ax = np.array(xs)
        
        n_x = normalize(ax)
        n_y = normalize(ay)
        
        if n_y[-1] == 1 or n_y[-1] == 0:
            continue

        diff_curve = n_y - n_x
            
        if diff_curve[-1] < np.max(diff_curve) * frac:
            break

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys


def optimize_diff_curve(f, n, steps=6, drop_off=0.5, frac=0.99, debug=False):
    b = n

    best_x = None
    best_y = 0

    while True:
        x, y = trace_diff_curve(f, 0, b, steps, drop_off)
        if debug: print(x, y)
        b = x[-2]
        if (y[:-1] < best_y*frac).all():
            if debug: print(y[:-1], best_y)
            break

        max_y = np.max(y)
        if best_y*frac < max_y:
            best_x = x[np.argmax(y)]
            best_y = max_y
            if debug: print(f"new best {max_y} {best_x}")

    return {'x': best_x, 'y': best_y}