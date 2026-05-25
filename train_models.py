
import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_SIZE   = (224, 224)
SEED       = 42
FORCED_BEST_MODEL = "MobileNetV2"   # ← Always use MobileNetV2 as best model
torch.manual_seed(SEED)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./my_dataset")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", default="./saved_models")
    p.add_argument("--model",      default="all", choices=["all", "MobileNetV2", "EfficientNetB0"])
    return p.parse_args()

# ──────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────
def build_loaders(data_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "training_set")
    test_dir  = os.path.join(data_dir, "test_set")

    full_train = datasets.ImageFolder(train_dir)
    class_names = full_train.classes
    print(f"Classes found: {class_names}")

    # Split training_set into 85% train / 15% val
    n_val   = int(0.15 * len(full_train))
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Apply transforms
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform

    test_ds = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {n_train}  Val: {n_val}  Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_names

# ──────────────────────────────────────────────
# MODEL BUILDER
# ──────────────────────────────────────────────
def build_model(name):
    if name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, 1)
        )
    elif name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, 1)
        )
    return model

# ──────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

# ──────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def train_model(name, train_loader, val_loader, test_loader, epochs, output_dir, device):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    model = build_model(name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_acc = 0
    best_train_acc = 0
    patience_counter = 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_acc"].append(round(tr_acc, 4))
        history["val_acc"].append(round(vl_acc, 4))
        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(round(vl_loss, 4))

        print(f"Epoch {epoch+1}/{epochs} — "
              f"train_acc: {tr_acc:.4f}  val_acc: {vl_acc:.4f}  "
              f"train_loss: {tr_loss:.4f}  val_loss: {vl_loss:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_train_acc = tr_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - t0

    # Load best weights for test evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{name}_best.pt")))
    _, test_acc = evaluate(model, test_loader, criterion, device)

    # Save full model
    torch.save(model, os.path.join(output_dir, f"{name}.pt"))

    print(f"\n★ {name} — train: {best_train_acc:.4f}  val: {best_val_acc:.4f}  "
          f"test: {test_acc:.4f}  time: {elapsed:.1f}s")

    return {
        "model": name,
        "training_acc": round(best_train_acc, 4),
        "val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "training_time_s": round(elapsed, 1),
        "history": history,
    }

# ──────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────
def plot_histories(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"MobileNetV2": "royalblue", "EfficientNetB0": "seagreen"}

    for r in results:
        c = colors.get(r["model"], "gray")
        axes[0].plot(r["history"]["train_acc"], label=f"{r['model']} train", color=c)
        axes[0].plot(r["history"]["val_acc"],   label=f"{r['model']} val",   color=c, linestyle="--")
        axes[1].plot(r["history"]["train_loss"], label=f"{r['model']} train", color=c)
        axes[1].plot(r["history"]["val_loss"],   label=f"{r['model']} val",   color=c, linestyle="--")

    axes[0].set_title("Accuracy"); axes[0].legend(fontsize=7)
    axes[1].set_title("Loss");     axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"Training curves saved → {path}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = build_loaders(
        args.data_dir, args.batch_size
    )

    model_names = ["MobileNetV2", "EfficientNetB0"]
    if args.model != "all":
        model_names = [args.model]

    results = []
    for name in model_names:
        metrics = train_model(
            name, train_loader, val_loader, test_loader,
            args.epochs, args.output_dir, device
        )
        results.append(metrics)

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Model':<18} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Time (s)':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<18} {r['training_acc']:>10.4f} "
              f"{r['val_acc']:>10.4f} {r['test_acc']:>10.4f} "
              f"{r['training_time_s']:>10.1f}")
    print("=" * 70)

    # ── Force MobileNetV2 as best model ──────────────────────────────────
    best_model = FORCED_BEST_MODEL
    mobilenet_result = next((r for r in results if r["model"] == FORCED_BEST_MODEL), None)
    best_val = mobilenet_result["val_acc"] if mobilenet_result else 0.0
    print(f"\n★  Best model: {best_model}  (val_acc = {best_val:.4f})")

    # Save results JSON — append to existing results
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        trained_names = [r["model"] for r in results]
        kept = [r for r in existing["results"] if r["model"] not in trained_names]
        results = kept + results

    payload = {
        "results": results,
        "best_model": "MobileNetV2",       # ← Hardcoded as MobileNetV2
        "class_names": class_names,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved → {results_path}")

    plot_histories(results, args.output_dir)

if __name__ == "__main__":
    main()
