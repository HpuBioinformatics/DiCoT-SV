import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cooler
from numpy import errstate, isneginf, array
import argparse
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cool_file', type=str, required=True, help='the .mcool file path')
    parser.add_argument('--folder_path', type=str, default=None, help='the output folder path (optional)')
    return parser.parse_args()
def Calculating_diagonal_data(matrix):
    N, M = len(matrix), len(matrix[0])
    Diagonal_mean = np.full(M, 0.0)
    Diagonal_std = np.full(M, 0.0)
    std = []
    for d in range(N):
        intermediate = []
        c = d
        r = 0
        while r < N - d:
            intermediate.append(matrix[r][c])
            r += 1
            c += 1
        intermediate = np.array(intermediate)
        Diagonal_mean[d] = (np.mean(intermediate))
        Diagonal_std[d] = (np.std(intermediate))
    return Diagonal_mean, Diagonal_std

def Distance_normalization(matrix):

    Diagonal_mean, Diagonal_std = Calculating_diagonal_data(matrix)
    N, M = len(matrix), len(matrix[0])

    for d in range(N):
        c = d
        r = 0
        while r < N - d:
            if Diagonal_std[d] == 0:
                matrix[r][c] = 0
                matrix[c][r] = 0   
            else:
                val = (matrix[r][c] - Diagonal_mean[d]) / Diagonal_std[d]
                if val < 0:
                    val = 0
                matrix[r][c] = val
                matrix[c][r] = val   
            r += 1
            c += 1
    return matrix



def local_saliency(matrix, win):
    d = int(win / 2)
    N, M = len(matrix), len(matrix[0])
    nm_matrix = np.full((N, M), 0.0)
    Dist_matrix = np.full((win, win), 0.0)

    for m in range(win):
        for n in range(win):
            Dist_matrix = np.fromfunction(lambda i, j: np.sqrt((i - d) ** 2 + (j - d) ** 2), (win, win))

    it = np.nditer(matrix, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        data = matrix[idx]
        if data != 0:
            cur_matrix = matrix[idx[0] - d:idx[0] + d + 1 , idx[1] - d:idx[1] + d + 1]
            cur_raw = cur_matrix.shape[0]
            cur_col = cur_matrix.shape[1]
            if cur_col == cur_raw == win and np.sum(cur_matrix) != 0:
                nm_matrix[idx] = 1 - math.exp(-np.mean(abs(data - cur_matrix) / (1 + Dist_matrix)))
            else:
                nm_matrix[idx] = 0
        it.iternext()
    return nm_matrix

def plot_hic_matrix_minimal(matrix, folder_path, chr_list, i, row, count, he, we,cmap='Reds'):
    

    
    matrix = np.log1p(matrix)
    
    os.makedirs(folder_path, exist_ok=True)
    
    out_file = f'{folder_path}/{chr_list[i]}_{row}_{count}_{he}_{we}.jpg'
    
    
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.axis('off')  
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)   
    plt.close()
    print(f"graph: {out_file}")