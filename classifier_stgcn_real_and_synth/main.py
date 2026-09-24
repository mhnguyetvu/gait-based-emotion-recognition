"""Controlled real + synthetic E-Gait Baseline-STEP experiment.

The fixed real-only 7:2:1 split is reused. A conditional STEP CVAE is fitted
only on real training gaits; validation and test sets remain real-only.
"""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime

import numpy as np
import torch


class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def distribution(labels, emotion_names):
    counts = Counter(np.asarray(labels, dtype=np.int64).tolist())
    return {
        str(class_id): {
            "emotion": emotion_names[class_id],
            "count": int(counts.get(class_id, 0)),
        }
        for class_id in sorted(emotion_names)
    }


def create_data_loader(dataset_module, data, labels, batch_size, shuffle, workers=0):
    dataset = dataset_module.TrainTestLoader(data, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
    )


def build_parser(script_dir, repo_dir):
    parser = argparse.ArgumentParser(
        description="Train Baseline-STEP with training-only CVAE augmentation."
    )
    parser.add_argument("--data-dir", default=os.path.join(repo_dir, "E-Gait"))
    parser.add_argument("--feature-file", default="features_merged.h5")
    parser.add_argument("--label-file", default="labels_merged.h5")
    parser.add_argument("--expected-samples", type=int, default=2177)
    parser.add_argument(
        "--work-dir",
        default=os.path.join(script_dir, "model_classifier_stgcn", "egait_2177_train_only_cvae_aug"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", default="Adam", choices=("Adam", "SGD"))
    parser.add_argument("--base-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--tiny-epochs", type=int, default=1000)
    parser.add_argument("--tiny-lr", type=float, default=0.01)
    parser.add_argument("--tiny-min-accuracy", type=float, default=100.0)
    parser.add_argument("--cvae-epochs", type=int, default=100)
    parser.add_argument("--cvae-batch-size", type=int, default=16)
    parser.add_argument("--cvae-lr", type=float, default=0.001)
    parser.add_argument("--cvae-beta", type=float, default=0.01)
    parser.add_argument("--synth-per-class", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_dir = os.path.dirname(script_dir)
    real_only_dir = os.path.join(repo_dir, "classifier_stgcn_real_only")
    # Reuse exactly the validated real-only loader, ST-GCN and processor.
    sys.path.insert(0, real_only_dir)
    from utils import loader, processor

    args = build_parser(script_dir, repo_dir).parse_args()
    args.print_log = not args.quiet
    args.save_log = True
    args.start_epoch = 0
    args.nesterov = True
    args.eval_batch_stats = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.abspath(args.work_dir), timestamp)
    generator_dir = os.path.join(run_dir, "train_only_cvae")
    tiny_dir = os.path.join(run_dir, "tiny_overfit")
    classifier_dir = os.path.join(run_dir, "full_classifier")
    log_dir = os.path.join(script_dir, "logs")
    for path in (run_dir, generator_dir, tiny_dir, classifier_dir, log_dir):
        os.makedirs(path, exist_ok=True)
    log_path = os.path.join(log_dir, "step_egait_2177_real_and_synth_{}.log".format(timestamp))

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    metadata_path = os.path.join(run_dir, "preprocessing_and_run_config.json")

    try:
        set_seed(args.seed)
        device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda:0"
        section("E-GAIT BASELINE-STEP WITH TRAINING-ONLY SYNTHETIC AUGMENTATION")
        print("Python version      :", sys.version.split()[0])
        print("PyTorch version     :", torch.__version__)
        print("CUDA available      :", torch.cuda.is_available())
        print("Selected device     :", device)
        print("Work directory      :", run_dir)
        print("Log file            :", log_path)

        section("LOAD AND VALIDATE THE 2,177 REAL GAITS")
        data, labels, source = loader.load_merged_egait(
            os.path.abspath(args.data_dir),
            feature_filename=args.feature_file,
            label_filename=args.label_file,
            target_frames=75,
            expected_samples=args.expected_samples if args.expected_samples > 0 else None,
        )
        print("Feature file        :", source["feature_path"])
        print("Label file          :", source["label_path"])
        print("Matched keys        :", len(source["keys"]))
        print("Emotion names       :", source["emotion_names"])
        print("Preprocessed shape  :", data.shape, "(N,T,J,C)")
        print("Preprocessing       : verified labels; canonical 16-joint order; root-center; linear T=75")

        section("16-SAMPLE TINY OVERFIT CHECK")
        tiny_indices = loader.make_tiny_balanced_indices(labels, samples_per_class=4, seed=args.seed)
        tiny_raw = data[tiny_indices]
        tiny_labels = labels[tiny_indices]
        tiny_normalizer = loader.CoordinateNormalizer().fit(tiny_raw)
        tiny_data = tiny_normalizer.transform(tiny_raw)
        tiny_args = copy.copy(args)
        tiny_args.work_dir = tiny_dir
        tiny_args.num_epoch = args.tiny_epochs
        tiny_args.base_lr = args.tiny_lr
        tiny_args.weight_decay = 0.0
        tiny_args.eval_interval = 1
        tiny_args.grad_clip = 0.0
        tiny_loader = create_data_loader(
            loader, tiny_data, tiny_labels, min(args.batch_size, len(tiny_data)), True
        )
        tiny_processor = processor.Processor(
            tiny_args,
            {"train": tiny_loader},
            in_channels=3,
            num_classes=4,
            graph_dict={"strategy": "spatial"},
            device=device,
            verbose=True,
        )
        tiny_result = tiny_processor.train_tiny_overfit(
            target_accuracy=args.tiny_min_accuracy,
            max_epochs=args.tiny_epochs,
        )
        print("Tiny-overfit result :", tiny_result)
        if not tiny_result["passed"]:
            raise RuntimeError("Tiny overfit failed; stopping before generator/full classifier training.")
        del tiny_processor, tiny_loader
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        section("REUSE THE FIXED STRATIFIED 7:2:1 REAL SPLIT")
        train_indices, val_indices, test_indices = loader.stratified_split_7_2_1(labels, seed=args.seed)
        train_raw, val_raw, test_raw = data[train_indices], data[val_indices], data[test_indices]
        train_labels, val_labels, test_labels = labels[train_indices], labels[val_indices], labels[test_indices]
        if set(train_indices) & set(val_indices) or set(train_indices) & set(test_indices) or set(val_indices) & set(test_indices):
            raise AssertionError("Real train/validation/test split indices overlap.")
        print("Real train/val/test : {}/{}/{}".format(len(train_indices), len(val_indices), len(test_indices)))
        print("Train labels        :", distribution(train_labels, source["emotion_names"]))
        print("Validation labels   :", distribution(val_labels, source["emotion_names"]))
        print("Test labels         :", distribution(test_labels, source["emotion_names"]))

        normalizer = loader.CoordinateNormalizer().fit(train_raw)
        train_data = normalizer.transform(train_raw)
        val_data = normalizer.transform(val_raw)
        test_data = normalizer.transform(test_raw)

        split = {
            "seed": int(args.seed),
            "train": int(len(train_indices)),
            "validation": int(len(val_indices)),
            "test": int(len(test_indices)),
            "train_indices": [int(i) for i in train_indices],
            "validation_indices": [int(i) for i in val_indices],
            "test_indices": [int(i) for i in test_indices],
            "train_sample_keys": [source["keys"][int(i)] for i in train_indices],
            "validation_sample_keys": [source["keys"][int(i)] for i in val_indices],
            "test_sample_keys": [source["keys"][int(i)] for i in test_indices],
        }
        split_path = os.path.join(run_dir, "real_split_manifest.json")
        with open(split_path, "w", encoding="utf-8") as stream:
            json.dump(split, stream, indent=2)

        training_npz = os.path.join(generator_dir, "real_training_split.npz")
        np.savez_compressed(training_npz, data=train_data, labels=train_labels)
        section("FIT CVAE USING ONLY REAL TRAINING GAITS")
        generator_script = os.path.join(repo_dir, "generator_cvae", "train_split_cvae.py")
        cvae_command = [
            sys.executable,
            generator_script,
            "--input-npz", training_npz,
            "--output-dir", generator_dir,
            "--epochs", str(args.cvae_epochs),
            "--batch-size", str(args.cvae_batch_size),
            "--samples-per-class", str(args.synth_per_class),
            "--learning-rate", str(args.cvae_lr),
            "--beta", str(args.cvae_beta),
            "--seed", str(args.seed),
            "--device", device,
        ]
        print("CVAE training samples: {} real train gaits only".format(len(train_data)))
        cvae_process = subprocess.Popen(
            cvae_command,
            cwd=os.path.join(repo_dir, "generator_cvae"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for output_line in iter(cvae_process.stdout.readline, ""):
            print(output_line, end="", flush=True)
        cvae_process.stdout.close()
        cvae_return_code = cvae_process.wait()
        if cvae_return_code:
            raise subprocess.CalledProcessError(cvae_return_code, cvae_command)
        synthetic_path = os.path.join(generator_dir, "synthetic_train_only.npz")
        with np.load(synthetic_path) as archive:
            synthetic_data = np.asarray(archive["data"], dtype=np.float32)
            synthetic_labels = np.asarray(archive["labels"], dtype=np.int64)
        expected_synthetic = 4 * args.synth_per_class
        if synthetic_data.shape != (expected_synthetic, 75, 16, 3):
            raise ValueError("Unexpected synthetic array shape: {}".format(synthetic_data.shape))
        if set(np.unique(synthetic_labels).tolist()) != {0, 1, 2, 3}:
            raise ValueError("Generated synthetic data does not contain all emotion classes.")
        print("Generated data      :", synthetic_data.shape)
        print("Synthetic labels    :", distribution(synthetic_labels, source["emotion_names"]))
        print("Generator provenance: trained only on the fixed real training split; held-out real gaits untouched")

        run_metadata = {
            "dataset": "E-Gait merged 2,177 real gaits + training-split-only generated gaits",
            "feature_file": source["feature_path"],
            "label_file": source["label_path"],
            "num_real_samples": int(len(data)),
            "label_map": {str(k): v for k, v in source["emotion_names"].items()},
            "source_lengths": {"min": int(min(source["source_lengths"])), "max": int(max(source["source_lengths"]))},
            "target_frames": 75,
            "source_joint_order": source["source_joint_order"],
            "model_joint_order": source["model_joint_order"],
            "joint_reorder_indices": source["joint_reorder_indices"],
            "coordinate_normalization": "root-center per frame; divide real and generated gaits by real-training-only xyz RMS",
            "coordinate_rms_scale": float(normalizer.scale),
            "split": split,
            "validation_and_test_are_real_only": True,
            "generator": {
                "implementation": "STEP conditional ST-GCN CVAE from generator_cvae/net/CVAE_stgcn.py",
                "fit_input": training_npz,
                "fit_scope": "real train split only",
                "validation_or_test_opened": False,
                "synthetic_samples": int(len(synthetic_data)),
                "synthetic_samples_per_class": int(args.synth_per_class),
                "metadata_file": os.path.join(generator_dir, "cvae_metadata.json"),
                "synthetic_file": synthetic_path,
                "loss": "mean-normalized position, anchor, velocity, acceleration and KL terms; see cvae_metadata.json",
            },
            "model": {"channels": [32, 64, 64], "temporal_kernel": 9, "input_shape": [3, 75, 16, 1], "classes": source["emotion_names"]},
            "training": {
                "epochs": int(args.num_epoch),
                "batch_size": int(args.batch_size),
                "optimizer": args.optimizer,
                "learning_rate": float(args.base_lr),
                "weight_decay": float(args.weight_decay),
                "lr_decay_epochs": [50, 75, 88] if args.num_epoch == 100 else [int(np.ceil(args.num_epoch * p)) for p in (0.50, 0.75, 0.876)],
                "real_training_gaits": int(len(train_data)),
                "synthetic_training_gaits": int(len(synthetic_data)),
                "total_training_gaits": int(len(train_data) + len(synthetic_data)),
            },
            "tiny_overfit": tiny_result,
            "real_only_reference": os.path.join(
                repo_dir,
                "classifier_stgcn_real_only",
                "model_classifier_stgcn",
                "egait_2177_baseline",
                "20260924_164111",
            ),
        }
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(run_metadata, stream, indent=2, sort_keys=True)
        print("Run metadata        :", metadata_path)
        print("Split manifest      :", split_path)

        section("TRAIN CLASSIFIER ON REAL TRAIN + TRAIN-ONLY SYNTHETIC")
        combined_train_data = np.concatenate((train_data, synthetic_data), axis=0)
        combined_train_labels = np.concatenate((train_labels, synthetic_labels), axis=0)
        train_loader = create_data_loader(loader, combined_train_data, combined_train_labels, args.batch_size, True, args.num_worker)
        val_loader = create_data_loader(loader, val_data, val_labels, args.batch_size, False, args.num_worker)
        test_loader = create_data_loader(loader, test_data, test_labels, args.batch_size, False, args.num_worker)
        full_args = copy.copy(args)
        full_args.work_dir = classifier_dir
        full_processor = processor.Processor(
            full_args,
            {"train": train_loader, "val": val_loader, "test": test_loader},
            in_channels=3,
            num_classes=4,
            graph_dict={"strategy": "spatial"},
            device=device,
            verbose=True,
        )
        metrics = full_processor.train()
        run_metadata["results"] = {
            "best_epoch": int(metrics["best_epoch"]),
            "train_accuracy_including_synthetic": float(metrics["train_accuracy"]),
            "validation_accuracy_real_only": float(metrics["best_val_accuracy"]),
            "test_accuracy_real_only": float(metrics["test_accuracy"]),
            "test_loss_real_only": float(metrics["test_loss"]),
        }
        run_metadata["real_only_reference_results"] = {
            "run": run_metadata["real_only_reference"],
            "validation_accuracy": 81.34,
            "test_accuracy": 80.54,
            "test_samples": 221,
        }
        run_metadata["delta_vs_real_only_percentage_points"] = {
            "validation_accuracy": round(float(metrics["best_val_accuracy"]) - 81.34, 2),
            "test_accuracy": round(float(metrics["test_accuracy"]) - 80.54, 2),
        }
        run_metadata["classifier_checkpoint"] = full_processor.best_checkpoint
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(run_metadata, stream, indent=2, sort_keys=True)

        section("REAL + SYNTHETIC RESULT")
        print("Best epoch                :", metrics["best_epoch"])
        print("Train Top1 (real+synth)   : {:.2f}%".format(metrics["train_accuracy"]))
        print("Validation Top1 (real)   : {:.2f}%".format(metrics["best_val_accuracy"]))
        print("Test Top1 (real)         : {:.2f}%".format(metrics["test_accuracy"]))
        print("Test loss (real)         : {:.6f}".format(metrics["test_loss"]))
        print("Real-only test reference : 80.54%")
        print("Log saved to              :", log_path)
        print("Run metadata              :", metadata_path)
    except Exception as exc:
        section("RUN FAILED")
        print("Exception type      :", type(exc).__name__)
        print("Exception message   :", exc)
        print("Log saved to        :", log_path)
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
