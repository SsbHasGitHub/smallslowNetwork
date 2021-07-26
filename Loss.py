import numpy as np
#记得转置矩阵，记得转置矩阵
#记得转置矩阵，记得转置矩阵
def crossEntropy( pre ,targets):
    return (pre-targets).T
def MSE(pre ,targets):
    return (pre-targets).T