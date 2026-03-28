#!/usr/bin/env python3
"""Test official HuggingFace example vs our approach."""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from torchvision import datasets
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# ---- Test 1: Official example approach (padding="max_length") ----
print("=== Test 1: Official approach (padding='max_length') ===")
cifar = datasets.CIFAR10(root="./data", train=False, download=True)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
texts = [f"a photo of a {c}" for c in classes]

correct = 0
total = 100
for i in range(total):
    image, label = cifar[i]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    # Official approach: process together, padding="max_length"
    inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits_per_image).squeeze()
    pred = probs.argmax().item()

    if pred == label:
        correct += 1

    if i < 5:
        print(f"  Sample {i}: true={classes[label]}, pred={classes[pred]}, "
              f"top prob={probs[pred]:.4f}, true prob={probs[label]:.4f}")

print(f"Official approach accuracy: {correct}/{total} = {100*correct/total:.1f}%\n")

# ---- Test 2: Our approach (padding=True, separate processing) ----
print("=== Test 2: Our approach (padding=True, separate) ===")
correct2 = 0
# Pre-compute text features with padding=True
text_inputs_ours = processor.tokenizer(texts, padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    text_feat = model.get_text_features(**text_inputs_ours)
    text_feat = F.normalize(text_feat, p=2, dim=-1)

for i in range(total):
    image, label = cifar[i]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    img_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_feat = model.get_image_features(**img_inputs)
        img_feat = F.normalize(img_feat, p=2, dim=-1)

    sims = (img_feat @ text_feat.T).squeeze()
    pred = sims.argmax().item()

    if pred == label:
        correct2 += 1

    if i < 5:
        print(f"  Sample {i}: true={classes[label]}, pred={classes[pred]}, "
              f"sims range=[{sims.min():.4f}, {sims.max():.4f}]")

print(f"Our approach accuracy: {correct2}/{total} = {100*correct2/total:.1f}%\n")

# ---- Test 3: Our approach but with padding="max_length" ----
print("=== Test 3: Our approach + padding='max_length' ===")
correct3 = 0
text_inputs_ml = processor.tokenizer(texts, padding="max_length", return_tensors="pt").to(device)

with torch.no_grad():
    text_feat_ml = model.get_text_features(**text_inputs_ml)
    text_feat_ml = F.normalize(text_feat_ml, p=2, dim=-1)

print(f"Text features cosine sim (padding=True vs max_length): "
      f"{F.cosine_similarity(text_feat[:1], F.normalize(model.get_text_features(**text_inputs_ours)[:1], dim=-1)).item():.6f}")

for i in range(total):
    image, label = cifar[i]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    img_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_feat = model.get_image_features(**img_inputs)
        img_feat = F.normalize(img_feat, p=2, dim=-1)

    sims = (img_feat @ text_feat_ml.T).squeeze()
    pred = sims.argmax().item()

    if pred == label:
        correct3 += 1

    if i < 5:
        print(f"  Sample {i}: true={classes[label]}, pred={classes[pred]}, "
              f"sims range=[{sims.min():.4f}, {sims.max():.4f}]")

print(f"padding='max_length' accuracy: {correct3}/{total} = {100*correct3/total:.1f}%")
