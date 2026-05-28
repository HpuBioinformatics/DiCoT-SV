import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import argparse

# ===============================
# 参数
# ===============================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='the model saved path')
    parser.add_argument('--candicate', default=None, help='the candicate region saved path')
    parser.add_argument('--output', default=None, help='the output path')
    return parser.parse_args()


# =====================================================
# === 修改点 1：替换模型定义（不再使用 LSTM）
# =====================================================
from CNN_attention_model import CNN_Attention_Classifier


def load_model(model_path, device):
    model = CNN_Attention_Classifier(num_classes=9)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# =====================================================
# 主逻辑
# =====================================================
args = get_args()
model_saved_path = args.model
candicate_saved_path = args.candicate
output_path = args.output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = load_model(model_saved_path, device)

candicate_save_path = candicate_saved_path

filenames = []
all_data = []
sv_list = []


# ===============================
# 计数函数（SV 后处理，保持不变）
# ===============================
def cal_counts(mat, type):
    if type == 1:
        return np.sum(mat[:10, :10])
    elif type == 2:
        return np.sum(mat[:10, 10:])
    elif type == 3:
        return np.sum(mat[10:, :10])
    elif type == 4:
        return np.sum(mat[10:, 10:])


# =====================================================
# === 修改点 2：读取数据（不再做 min-max）
# =====================================================
for filename in os.listdir(candicate_save_path):
    if not filename.endswith(".txt"):
        continue

    txt_path = os.path.join(candicate_save_path, filename)
    matrix = np.loadtxt(txt_path)

    if matrix.shape != (20, 20):
        continue

    filenames.append(filename)
    all_data.append(matrix.astype(np.float32))


# =====================================================
# 推理 + SV 后处理（保持原样）
# =====================================================
for i in range(len(all_data)):
    x = torch.tensor(all_data[i]).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,20,20)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        prob, predicted_class = torch.max(probs, 1)

    prob = prob.item()
    predicted_class = predicted_class.item()
     #目前最好的一版阈值是0.98 /mnt/sdc/chenpei/Hi-C/DEIM_output_n_2/find_region/predict.thrh/0.98and0.98
    # if (predicted_class in [1, 2, 3, 4, 7, 8] and prob > 0.98) or \
    #    (predicted_class in [5, 6] and prob > 0.98):
    if (predicted_class in [1, 2, 3, 4, 7, 8] and prob > 0.97) or \
       (predicted_class in [5, 6] and prob > 0.97):   
        name = filenames[i].split('.')[0]
        parts = name.split('_')
        local1 = int(name.split('_')[2])*50000
        local2 = int(name.split('_')[3])*50000
        print(name,local1,local2,prob)
        # ================= inter-chr =================
        if parts[1][0].isalpha():
            chr1 = parts[0]
            chr2 = parts[1]
            pos1 = int(parts[2])
            pos2 = int(parts[3])

            flag = True
            for j in range(len(sv_list)):
                if chr1 == sv_list[j][0] and chr2 == sv_list[j][1] and \
                   abs(sv_list[j][2] - pos1) < 10 and abs(sv_list[j][3] - pos2) < 10:
                    flag = False
                if chr1 == sv_list[j][0] and chr2 == sv_list[j][1] and \
                   abs(sv_list[j][3] - pos1) < 3 and abs(sv_list[j][2] - pos2) < 3:
                    flag = False

            if flag:
                sv_list.append([chr1, chr2, pos1, pos2, predicted_class])

        # ================= intra-chr =================
        else:
            chr = parts[0]
            pos1 = int(parts[1])
            pos2 = int(parts[2])

            if pos1 > pos2:
                pos1, pos2 = pos2, pos1

            flag = True
            for j in range(len(sv_list)):
                if chr == sv_list[j][0] and \
                   abs(sv_list[j][1] - pos1) < 10 and abs(sv_list[j][2] - pos2) < 10:
                    flag = False
                    continue

                if chr == sv_list[j][0] and \
                   abs(sv_list[j][2] - pos1) < 3 and abs(sv_list[j][1] - pos2) < 3:
                    flag = False
                    continue

                if chr == sv_list[j][0] and \
                   abs(sv_list[j][1] - pos1) < 5 and abs(sv_list[j][2] - pos2) <= 15:

                    mat1 = all_data[i]
                    mat2 = np.loadtxt(sv_list[j][4])

                    if cal_counts(mat1, predicted_class) > cal_counts(mat2, predicted_class):
                        sv_list[j][1] = pos1
                        sv_list[j][2] = pos2
                    else:
                        flag = False

            if flag:
                sv_list.append([
                    chr, pos1, pos2,
                    predicted_class,
                    os.path.join(candicate_save_path, filenames[i])
                ])


# =====================================================
# 输出结果（完全保持原样）
# =====================================================
os.makedirs(output_path, exist_ok=True)

if len(sv_list) == 0:
    with open(os.path.join(output_path, 'SV_list.txt'), 'w') as f:
        pass

else:
    if isinstance(sv_list[0][1], str):  # inter
        with open(os.path.join(output_path, 'Inter_SV_list.txt'), 'w') as f:
            f.write('chrA\tbreakpoint_A\tchrB\tbreakpoint_B\tSV_type\n')
            for item in sv_list:
                f.write(
                    f"{item[0]}\t{item[2]*50000}\t{item[1]}\t{item[3]*50000}\t{item[4]}\n"
                )
    else:
        with open(os.path.join(output_path, 'Intra_SV_list.txt'), 'w') as f:
            f.write('chr\tbreakpoint_A\tbreakpoint_B\tSV_type\n')
            for item in sv_list:
                f.write(
                    f"{item[0]}\t{item[1]*50000}\t{item[2]*50000}\t{item[3]}\n"
                )
