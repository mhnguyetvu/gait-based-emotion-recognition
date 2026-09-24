"""Fit the STEP CVAE on a caller-provided training split and sample gaits.

The input archive must contain only classifier-training gaits, already
preprocessed and scaled with training-only statistics. Held-out samples are
never opened by this script.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

GENERATOR_DIR = os.path.dirname(os.path.realpath(__file__))
if GENERATOR_DIR in sys.path:
    sys.path.remove(GENERATOR_DIR)
sys.path.insert(0, GENERATOR_DIR)
# The legacy venv also exposes classifier_stgcn_real_only/net as a top-level
# package. Ensure absolute `net.*` imports in this generator resolve locally.
for module_name in tuple(sys.modules):
    if module_name == "net" or module_name.startswith("net."):
        module = sys.modules[module_name]
        module_file = getattr(module, "__file__", "") or ""
        if not os.path.abspath(module_file).startswith(os.path.abspath(GENERATOR_DIR)):
            del sys.modules[module_name]

from net.CVAE_stgcn import CVAE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sequence_loss(target, reconstruction, mean, log_variance, beta):
    """Stable mean-normalized position, anchor, motion, and KL objective."""
    loss = F.mse_loss(reconstruction, target)
    for anchor in (0, target.shape[2] // 2, target.shape[2] - 1):
        loss = loss + F.mse_loss(
            reconstruction - reconstruction[:, :, anchor:anchor + 1],
            target - target[:, :, anchor:anchor + 1],
        )

    target_velocity = target[:, :, 1:] - target[:, :, :-1]
    output_velocity = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
    loss = loss + F.mse_loss(output_velocity, target_velocity)
    target_acceleration = target_velocity[:, :, 1:] - target_velocity[:, :, :-1]
    output_acceleration = output_velocity[:, :, 1:] - output_velocity[:, :, :-1]
    loss = loss + F.mse_loss(output_acceleration, target_acceleration)

    kl = -0.5 * torch.mean(
        1.0 + log_variance - mean.pow(2) - log_variance.exp()
    )
    return loss + beta * kl, loss.detach(), kl.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    with np.load(args.input_npz) as archive:
        data = np.asarray(archive["data"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)

    if data.ndim != 4 or data.shape[1:] != (75, 16, 3):
        raise ValueError("Expected training data shaped (N,75,16,3), got {}".format(data.shape))
    if labels.shape != (len(data),) or set(np.unique(labels).tolist()) != {0, 1, 2, 3}:
        raise ValueError("Expected one valid 0..3 emotion label per training gait.")
    if not np.isfinite(data).all():
        raise ValueError("CVAE training input contains NaN or Inf.")

    # CVAE input layout: N,C,T,V,M. It receives only the passed train split.
    tensor_data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2))[:, :, :, :, None].copy())
    tensor_labels = torch.from_numpy(labels)
    dataset = TensorDataset(tensor_data, tensor_labels)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = CVAE(
        in_channels=3,
        T=75,
        V=16,
        n_z=args.latent_dim,
        num_classes=4,
        graph_args={"strategy": "spatial"},
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print("CVAE device          :", device, flush=True)
    print("CVAE training gaits  :", len(dataset), "(train split only)", flush=True)
    print("CVAE architecture    : STEP ST-GCN encoder/decoder, latent={}".format(args.latent_dim), flush=True)
    print("CVAE epochs/batch    : {}/{}".format(args.epochs, args.batch_size), flush=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.float().to(device)
            batch_labels = batch_labels.long().to(device)
            one_hot = F.one_hot(batch_labels, num_classes=4).float()
            label_map = one_hot[:, :, None, None, None].expand(-1, -1, 75, 16, 1)

            optimizer.zero_grad(set_to_none=True)
            reconstruction, mean, log_variance, _ = model(batch_data, label_map, one_hot)
            loss, reconstruction_loss, kl_loss = sequence_loss(
                batch_data, reconstruction, mean, log_variance, args.beta
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CVAE loss at epoch {}.".format(epoch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append((float(loss.item()), float(reconstruction_loss.item()), float(kl_loss.item())))

        mean_loss = np.mean(losses, axis=0)
        history.append({
            "epoch": epoch + 1,
            "loss": float(mean_loss[0]),
            "reconstruction_loss": float(mean_loss[1]),
            "kl_loss": float(mean_loss[2]),
        })
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(
                "CVAE epoch {:3d}/{:3d} | total {:.6f} | reconstruction {:.6f} | KL {:.6f}".format(
                    epoch + 1, args.epochs, *mean_loss
                ),
                flush=True,
            )

    checkpoint_path = os.path.join(args.output_dir, "cvae_train_split.pt")
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    generated = []
    generated_labels = []
    with torch.no_grad():
        for class_id in range(4):
            remaining = args.samples_per_class
            while remaining:
                batch_count = min(args.batch_size, remaining)
                z = torch.randn(batch_count, args.latent_dim, device=device)
                class_labels = torch.full((batch_count,), class_id, dtype=torch.long, device=device)
                one_hot = F.one_hot(class_labels, num_classes=4).float()
                sample = model.decoder(z, one_hot, 75, 16)
                # All real inputs were root-centered. Apply the same constraint
                # to generated samples before classifier training.
                sample = sample - sample[:, :, :, 0:1, :]
                sample = sample[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()
                generated.append(sample.cpu().numpy().astype(np.float32))
                generated_labels.extend([class_id] * batch_count)
                remaining -= batch_count

    synthetic_data = np.concatenate(generated, axis=0)
    synthetic_labels = np.asarray(generated_labels, dtype=np.int64)
    if not np.isfinite(synthetic_data).all():
        raise FloatingPointError("Generated CVAE data contains NaN or Inf.")
    synthetic_path = os.path.join(args.output_dir, "synthetic_train_only.npz")
    np.savez_compressed(synthetic_path, data=synthetic_data, labels=synthetic_labels)
    metadata = {
        "generator": "generator_cvae/net/CVAE_stgcn.py (STEP conditional ST-GCN CVAE)",
        "generator_fit_samples": int(len(data)),
        "generator_fit_scope": "fixed classifier training split only",
        "held_out_data_opened": False,
        "synthetic_samples": int(len(synthetic_data)),
        "synthetic_samples_per_class": int(args.samples_per_class),
        "coordinate_scaling": "same train-fitted real-only RMS normalized coordinates",
        "root_constraint": "generated root subtracted per frame",
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "beta": float(args.beta),
        "latent_dim": int(args.latent_dim),
        "loss": "mean-normalized position MSE + three anchor-relative MSE terms + velocity MSE + acceleration MSE + beta*mean KL",
        "loss_note": "Stable mean-normalized form; differs in scale from the legacy between_frame_loss implementation.",
        "checkpoint": checkpoint_path,
        "synthetic_data": synthetic_path,
        "training_history": history,
    }
    with open(os.path.join(args.output_dir, "cvae_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print("Generated synthetic :", synthetic_data.shape, flush=True)
    print("Synthetic archive   :", synthetic_path, flush=True)
    print("CVAE checkpoint     :", checkpoint_path, flush=True)


if __name__ == "__main__":
    main()
