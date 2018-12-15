import numpy as np
def shuffle_data(var_list):
    length = var_list[0].shape[0]
    perm = np.random.permutation(length)
    ret = []
    for var in var_list:
        np.take(var,perm,axis=0,out=var)
        ret.append(var)
    return ret
