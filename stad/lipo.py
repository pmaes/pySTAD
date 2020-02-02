import numpy as np

# See:
# Malherbe, C., & Vayatis, N. (2017). Global optimization of Lipschitz functions. 34th International Conference on Machine Learning, ICML 2017, 5(1972), 3592–3601.
# https://github.com/UBC-CS/lipo-python/blob/master/src/sequential.py

def adaptive_lipo(func, bounds, n, p=0.1):
    # dimension of the domain
    d = len(bounds)

    alpha = 0.01/d
    k_seq=(1+alpha)**np.arange(-10000,10000) # Page 16

    # preallocate the output arrays
    y = np.zeros(n) - np.Inf
    x = np.zeros((n, d))
    loss = np.zeros(n)
    k_arr = np.zeros(n)
    
    # the lower/upper bounds on each dimension
    bound_mins = np.array([bnd[0] for bnd in bounds])
    bound_maxs = np.array([bnd[1] for bnd in bounds])
    
    # initialization with randomly drawn point in domain and k = 0
    k = 0
    k_est = -np.inf
    u = np.random.rand(d)
    x_prop = u * (bound_maxs - bound_mins) + bound_mins
    x[0] = x_prop
    y[0] = func(x_prop)
    k_arr[0] = k

    upper_bound = lambda x_prop, y, x, k: np.min(y+k*np.linalg.norm(x_prop-x,axis=1))

    for t in np.arange(1, n):
        # draw a uniformly distributed random variable
        u = np.random.rand(d)
        x_prop = u * (bound_maxs - bound_mins) + bound_mins

        # check if we are exploring or exploiting
        if np.random.rand() > p: # enter to exploit w/ prob (1-p)
            # exploiting - ensure we're drawing from potential maximizers
            while upper_bound(x_prop, y[:t], x[:t], k) < np.max(y[:t]):
                u = np.random.rand(d)
                x_prop = u * (bound_maxs - bound_mins) + bound_mins 
        else:
            pass 
            # we keep the randomly drawn point as our next iterate
            # this is "exploration"
        # add proposal to array of visited points
        x[t] = x_prop
        y[t] = func(x_prop)
        loss[t] = np.max(y)

        # compute current number of tracked distances
        old_num_dist = (t * (t - 1)) // 2
        new_num_dist = old_num_dist + t

        new_x_dist = np.sqrt(np.sum((x[:t] - x[t])**2, axis=1))

        new_y_dist = np.abs(y[:t] - y[t])

        k_est = max(k_est, np.max(new_y_dist/new_x_dist))  # faster
        k = k_seq[np.argmax(k_seq >= k_est)]
        i_t = np.ceil(np.log(k_est)/np.log(1+alpha))
        k = (1+alpha)**i_t
        k_arr[t] = k
        

    output = {'loss': loss, 'x': x, 'y': y, 'k': k_arr}
    return output
