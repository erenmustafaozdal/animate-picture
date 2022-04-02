import torch
from PIL import Image
import glob
import os
from pathlib import Path


model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)


# exports klasörünü sil
for filename in glob.glob("./exports/*"):
    os.remove(filename)


# images klasöründeki resimleri al ve çevir kaydet
for file in glob.glob("./images/*"):
    filename = Path(file).stem
    img = Image.open(file).convert("RGB")
    out = face2paint(model, img)
    out.save(f"./exports/{filename}.png")
