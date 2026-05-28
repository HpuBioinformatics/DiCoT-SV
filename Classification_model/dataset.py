# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class MatrixDataset(Dataset):
#     def __init__(self, root_dir):
#         self.samples = []

#         for label in range(9):  # 0-8
#             class_dir = os.path.join(root_dir, str(label))
#             for fname in os.listdir(class_dir):
#                 if fname.endswith(".txt"):
#                     self.samples.append(
#                         (os.path.join(class_dir, fname), label)
#                     )

#     def __len__(self):
#         return len(self.samples)

#     # def __getitem__(self, idx):
#     #     path, label = self.samples[idx]

#     #     matrix = np.loadtxt(path)        # (20, 20)
#     #     matrix = torch.tensor(matrix, dtype=torch.float32)
#     #     matrix = matrix.unsqueeze(0)     # (1, 20, 20)

#     #     return matrix, label
#     def __getitem__(self, idx):
#         path, label = self.samples[idx]

#         matrix = np.loadtxt(path)

#             # 强制处理为 20x20
#         h, w = matrix.shape

#         fixed = np.zeros((20, 20), dtype=np.float32)

#         h_use = min(h, 20)
#         w_use = min(w, 20)

#         fixed[:h_use, :w_use] = matrix[:h_use, :w_use]

#         matrix = torch.tensor(fixed, dtype=torch.float32)
#         matrix = matrix.unsqueeze(0)  # (1, 20, 20)

#         return matrix, label


import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MatrixDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label in range(9):
            class_dir = os.path.join(root_dir, str(label))
            for fname in os.listdir(class_dir):
                if fname.endswith(".txt"):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        mat = np.loadtxt(path)

        # 强制 pad / crop 到 20x20
        fixed = np.zeros((20, 20), dtype=np.float32)
        h, w = mat.shape
        fixed[:min(h,20), :min(w,20)] = mat[:20, :20]

        mat = torch.tensor(fixed, dtype=torch.float32)
        mat = mat.unsqueeze(0)  # (1, 20, 20)

        return mat, label


