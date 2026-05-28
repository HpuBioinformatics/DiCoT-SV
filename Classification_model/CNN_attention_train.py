import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from collections import Counter
import numpy as np

from dataset import MatrixDataset
from CNN_attention_model import CNN_Attention_Classifier



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_weights(dataset, num_classes=9):
    labels = [label for _, label in dataset]
    counter = Counter(labels)

    counts = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        counts[i] = counter.get(i, 0)

    counts[counts == 0] = 1.0

    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)



def evaluate(model, loader, device, num_classes=9):
    model.eval()

    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            for c in range(num_classes):
                TP[c] += ((preds == c) & (y == c)).sum().item()
                FP[c] += ((preds == c) & (y != c)).sum().item()
                FN[c] += ((preds != c) & (y == c)).sum().item()

    acc = correct / total
    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    macro_f1 = f1.mean()

    return acc, recall, macro_f1



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = MatrixDataset("/mnt/sdc/chenpei/Hi-C/CNN_Attention/train")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

   
    train_loader = DataLoader(
        train_set,
        batch_size=16,          
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

   
    model = CNN_Attention_Classifier(num_classes=9).to(device)

    class_weights = compute_class_weights(train_set).to(device)

    criterion = FocalLoss(
        alpha=class_weights,
        gamma=1.0
    )

    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,               
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100
    )

    epochs = 100
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        acc, recall, macro_f1 = evaluate(model, test_loader, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"Acc: {acc:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )
        print("Per-class Recall:", np.round(recall, 3))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                model.state_dict(),
                "cnn_attn_groupnorm_best_3.pth"
            )

    print("Training finished.")


if __name__ == "__main__":
    main()







