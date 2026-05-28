import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cooler
import argparse
from numpy import errstate, isneginf, array
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
            else:
                if matrix[r][c] - Diagonal_mean[d] < 0:
                    matrix[r][c] = 0
                else:
                    matrix[r][c] = (matrix[r][c] - Diagonal_mean[d]) / Diagonal_std[d]
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



def plot_hic_matrix_minimal(matrix, output_path, chr_list, i, j, row, count, he, we,cmap='Reds'):
    
    matrix = np.log1p(matrix)  
    os.makedirs(output_path, exist_ok=True)
    out_file = f'{output_path}/{chr_list[i]}_{chr_list[j]}_{row}_{count}_{he}_{we}.jpg'
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.axis('off')  

    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"graph: {output_path}")

def main():

    args = get_args()
    cool_file = args.cool_file
    print("Input file:", cool_file)

   
    cell_name = cool_file.split("/")[-1].split('.')[0]
    print("Cell name:", cell_name)

    resolution = 50000
    clr = cooler.Cooler(f'{cool_file}::resolutions/{resolution}')
    contact_matrix = clr.matrix(balance=False)

    
    chr_list = [
        'chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
        'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
        'chr20','chr21','chr22','chrX'
    ]

    
    
    
    if args.folder_path is not None:
        folder_path = args.folder_path
    else:
        folder_path = f'./{cell_name}_inter_image_{resolution}'

    print("Output folder:", folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    
    tile_size = 200
    overlap_percentage = 0.2
    overlap = int(tile_size * overlap_percentage)

    
    for i in range(len(chr_list)):
        for j in range(i+1, len(chr_list)):
            print("Processing:", chr_list[i], chr_list[j])
            chr_mat_one = contact_matrix.fetch(chr_list[i], chr_list[j])
            chr_mat = local_saliency(chr_mat_one,11)
            height, weight = chr_mat.shape
            print("Matrix size:", weight, height)

            x, y = 0, 0
            row = 0

            while y < height - 10:
                row += 1
                count = 0
                while x < weight - 10:
                    count += 1
                    sv_subregion = chr_mat[y:y+tile_size, x:x+tile_size]
                    if np.sum(sv_subregion > 0.001) > 3:
                        print(sv_subregion.shape)
                        he, we = sv_subregion.shape
                        plot_hic_matrix_minimal(sv_subregion, folder_path, chr_list, i, j, row, count, he,we,cmap='Reds')
                    x += tile_size - overlap
                x = 0
                y += tile_size - overlap

if __name__ == '__main__':
    main()