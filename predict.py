import os
import cv2
import torch
import yaml
from torchvision.ops import nms


from utils.config import build_model_from_cfg
from utils.checkpoint import load_checkpoint



def load_deim(cfg_path, ckpt_path, device="cuda"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

 
    model = build_model_from_cfg(cfg)
    model.to(device)


    load_checkpoint(model, ckpt_path)
    model.eval()

    return model



def infer_image(model, img_path, device="cuda", score_thr=0.5):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb).float().permute(2, 0, 1) / 255.
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs["boxes"][0].cpu()
    scores = outputs["scores"][0].cpu()

    keep = scores > score_thr
    boxes = boxes[keep]
    scores = scores[keep]

  
    idx = nms(boxes, scores, 0.5)
    return img, boxes[idx], scores[idx]



def draw_box(img, boxes, scores, save_path):
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{s:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(save_path, img)



def process_dir(img_dir, out_dir, cfg, ckpt, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)

    model = load_deim(cfg, ckpt, device)

    for name in os.listdir(img_dir):
        if not name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, name)
        save_path = os.path.join(out_dir, name)

        print("Processing:", img_path)
        img, boxes, scores = infer_image(model, img_path, device)
        draw_box(img, boxes, scores, save_path)

    print("Done! Results saved in:", out_dir)



if __name__ == "__main__":
    process_dir(
        img_dir="",
        out_dir="",
        cfg="configs/deim_dfine/deim_hgnetv2_n_coco.yml",
        ckpt="",
        device="cuda"
    )
