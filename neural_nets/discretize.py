import numpy as np




def scale_round(w,scale): # scale=10 okay for MLP, scale=100 okay for CNN
    return [np.round(w[i]*scale).astype(int) for i in range(len(w))]


def custom_scale(w,tolerance=3):
    new_weights = []
    candidate_scales = [5,10,20,30,40,50,60,70,80,90,100]
    n = len(w)
    too_many_zeros = lambda l : len(l)/l.count(0) < tolerance

    for i in range(int(round(n/2))):
        best_weights = []
        for s in candidate_scales:
            if not too_many_zeros(np.array(scale_round([w[2*i]+w[2*i+1]],s)).flatten().tolist()):
                best_weights = scale_round([w[2*i],w[2*i+1]],s)
                break
        if len(best_weights) == 0:
            best_weights = scale_round([w[2*i],w[2*i+1]],100)
        new_weights += best_weights
    return new_weights
