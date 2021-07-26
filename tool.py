import numpy as np
def score(y_true,y_pre):
    u=((y_true-y_pre)**2).sum()
    v=((y_true-y_true.mean())**2).sum()
    return 1-u/v

