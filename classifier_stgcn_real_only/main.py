"""Train the real-only STEP Baseline-STEP classifier on merged E-Gait."""

import argparse
import copy
import json
import os
import random
import sys
import traceback
from collections import Counter
from datetime import datetime

import numpy as np
import torch

from utils import loader, processor


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


def print_distribution(title, labels):
    print(title)
    counts = Counter(np.asarray(labels, dtype=np.int64).tolist())
    for class_id in sorted(loader.EMOTION_NAMES):
        print(
            "  {} ({}) : {}".format(
                class_id,
                loader.EMOTION_NAMES[class_id],
                counts.get(class_id, 0),
            )
        )


def describe_array(title, data):
    data = np.asarray(data)
    print(title)
    print("  shape :", data.shape)
    print("  dtype :", data.dtype)
    print("  finite:", bool(np.isfinite(data).all()))
    print("  min/max:", float(np.min(data)), float(np.max(data)))
    print("  mean/std:", float(np.mean(data)), float(np.std(data)))


def create_data_loader(data, labels, batch_size, shuffle, num_workers):
    dataset = loader.TrainTestLoader(data, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )


def build_parser(default_data_dir, default_model_dir):
    parser = argparse.ArgumentParser(
        description="Run the 2,177-real-gait E-Gait Baseline-STEP experiment."
    )
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--feature-file", default="features_merged.h5")
    parser.add_argument("--label-file", default="labels_merged.h5")
    parser.add_argument("--expected-samples", type=int, default=2177)
    parser.add_argument("--work-dir", default=default_model_dir)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epoch", "--num_epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", default="Adam", choices=("Adam", "SGD", "adam", "sgd"))
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
    parser.add_argument(
        "--tiny-only",
        action="store_true",
        help="Run only the required 16-sample overfit diagnostic.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_dir = os.path.dirname(script_dir)
    default_data_dir = os.path.join(repo_dir, "E-Gait")
    default_model_dir = os.path.join(
        script_dir, "model_classifier_stgcn", "egait_2177_baseline"
    )
    args = build_parser(default_data_dir, default_model_dir).parse_args()
    args.print_log = not args.quiet
    args.save_log = True
    args.start_epoch = 0
    args.nesterov = True
    args.eval_batch_stats = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.abspath(args.work_dir), timestamp)
    tiny_dir = os.path.join(run_dir, "tiny_overfit")
    full_dir = os.path.join(run_dir, "full_baseline")
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(tiny_dir, exist_ok=True)
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "step_egait_2177_{}.log".format(timestamp))

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        set_seed(args.seed)
        device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda:0"

        section("E-GAIT REAL-ONLY BASELINE-STEP")
        print("Python version      :", sys.version.split()[0])
        print("PyTorch version     :", torch.__version__)
        print("CUDA available      :", torch.cuda.is_available())
        print("Selected device     :", device)
        print("Data directory      :", os.path.abspath(args.data_dir))
        print("Work directory      :", run_dir)
        print("Log file            :", log_path)
        print("Full-run parameters : epochs={}, batch={}, optimizer={}, lr={}, weight_decay={}".format(
            args.num_epoch, args.batch_size, args.optimizer, args.base_lr, args.weight_decay
        ))
        print("Tiny diagnostic     : 16 balanced samples; target {:.2f}%; max {} epochs; lr={}".format(
            args.tiny_min_accuracy, args.tiny_epochs, args.tiny_lr
        ))

        section("LOAD AND VALIDATE MERGED DATA")
        data, labels, metadata = loader.load_merged_egait(
            os.path.abspath(args.data_dir),
            feature_filename=args.feature_file,
            label_filename=args.label_file,
            target_frames=75,
            expected_samples=args.expected_samples if args.expected_samples > 0 else None,
        )
        print("Feature file        :", metadata["feature_path"])
        print("Label file          :", metadata["label_path"])
        print("Emotion map         :", metadata["emotion_map_path"] or "built-in E-Gait map")
        print("Emotion labels      :", metadata["emotion_names"])
        print("Feature/label keys  :", len(metadata["keys"]), "matched exactly")
        print("Raw frame range     :", min(metadata["source_lengths"]), "to", max(metadata["source_lengths"]))
        print("Preprocessed shape  :", data.shape, "(N,T,J,C)")
        print("Source joint order  :", metadata["source_joint_order"])
        print("Model joint order   :", metadata["model_joint_order"])
        print("Joint reorder index :", metadata["joint_reorder_indices"])
        print("Temporal transform  : linear resampling to 75 frames")
        print("Coordinate transform: per-frame root centering; one train-fitted xyz RMS scale")
        print_distribution("Verified emotion label counts:", labels)
        describe_array("Root-centered, resampled coordinates before global scaling:", data)

        section("16-SAMPLE TINY OVERFIT TEST")
        tiny_indices = loader.make_tiny_balanced_indices(labels, samples_per_class=4, seed=args.seed)
        tiny_raw = data[tiny_indices]
        tiny_labels = labels[tiny_indices]
        tiny_normalizer = loader.CoordinateNormalizer().fit(tiny_raw)
        tiny_data = tiny_normalizer.transform(tiny_raw)
        print("Tiny samples        :", len(tiny_data))
        print("Tiny label counts   : one sample set with 4 examples per emotion")
        print("Tiny coordinate RMS : {:.9f}".format(tiny_normalizer.scale))
        describe_array("Normalized tiny training data:", tiny_data)

        tiny_args = copy.copy(args)
        tiny_args.work_dir = tiny_dir
        tiny_args.num_epoch = args.tiny_epochs
        tiny_args.base_lr = args.tiny_lr
        tiny_args.weight_decay = 0.0
        tiny_args.eval_interval = 1
        tiny_args.grad_clip = 0.0
        tiny_args.log_interval = max(1, args.log_interval)
        tiny_loader = create_data_loader(
            tiny_data,
            tiny_labels,
            batch_size=min(args.batch_size, len(tiny_data)),
            shuffle=True,
            num_workers=args.num_worker,
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
        print("Tiny overfit result :", tiny_result)
        if not tiny_result["passed"]:
            raise RuntimeError(
                "Tiny-overfit accuracy was {:.2f}% (target {:.2f}%); "
                "full training was stopped. Inspect the activation diagnostics."
                .format(tiny_result["train_accuracy"], args.tiny_min_accuracy)
            )
        if args.tiny_only:
            section("TINY-ONLY RUN COMPLETE")
            print("Tiny checkpoint     :", tiny_result["checkpoint"])
            print("Log saved to        :", log_path)
            return

        section("STRATIFIED 7:2:1 FULL-DATA SPLIT")
        train_indices, val_indices, test_indices = loader.stratified_split_7_2_1(
            labels, seed=args.seed
        )
        train_raw = data[train_indices]
        val_raw = data[val_indices]
        test_raw = data[test_indices]
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]
        test_labels = labels[test_indices]
        if set(train_indices) & set(val_indices) or set(train_indices) & set(test_indices) or set(val_indices) & set(test_indices):
            raise AssertionError("Train/validation/test splits overlap.")
        print("Train samples       :", len(train_indices))
        print("Validation samples  :", len(val_indices))
        print("Test samples        :", len(test_indices))
        print_distribution("Train emotion counts:", train_labels)
        print_distribution("Validation emotion counts:", val_labels)
        print_distribution("Test emotion counts:", test_labels)

        normalizer = loader.CoordinateNormalizer().fit(train_raw)
        train_data = normalizer.transform(train_raw)
        val_data = normalizer.transform(val_raw)
        test_data = normalizer.transform(test_raw)
        print("Shared train RMS scale: {:.9f}".format(normalizer.scale))
        describe_array("Normalized train data:", train_data)
        describe_array("Normalized validation data:", val_data)
        describe_array("Normalized test data:", test_data)

        run_metadata = {
            "dataset": "E-Gait merged real-only release",
            "feature_file": metadata["feature_path"],
            "label_file": metadata["label_path"],
            "num_samples": int(len(data)),
            "label_map": {str(k): v for k, v in metadata["emotion_names"].items()},
            "source_lengths": {
                "min": int(min(metadata["source_lengths"])),
                "max": int(max(metadata["source_lengths"])),
            },
            "target_frames": 75,
            "source_joint_order": metadata["source_joint_order"],
            "model_joint_order": metadata["model_joint_order"],
            "joint_reorder_indices": metadata["joint_reorder_indices"],
            "coordinate_normalization": "per-frame root-center; divide by train-set non-root xyz RMS",
            "coordinate_rms_scale": normalizer.scale,
            "split": {
                "seed": args.seed,
                "train": int(len(train_indices)),
                "validation": int(len(val_indices)),
                "test": int(len(test_indices)),
                "train_indices": [int(index) for index in train_indices],
                "validation_indices": [int(index) for index in val_indices],
                "test_indices": [int(index) for index in test_indices],
                "train_sample_keys": [metadata["keys"][int(index)] for index in train_indices],
                "validation_sample_keys": [metadata["keys"][int(index)] for index in val_indices],
                "test_sample_keys": [metadata["keys"][int(index)] for index in test_indices],
            },
            "model": {
                "channels": [32, 64, 64],
                "temporal_kernel": 9,
                "input_shape": [3, 75, 16, 1],
                "classes": metadata["emotion_names"],
            },
            "training": {
                "epochs": args.num_epoch,
                "batch_size": args.batch_size,
                "optimizer": args.optimizer,
                "learning_rate": args.base_lr,
                "weight_decay": args.weight_decay,
                "lr_decay_epochs": (
                    [250, 375, 438]
                    if args.num_epoch == 500
                    else [
                        int(np.ceil(args.num_epoch * 0.50)),
                        int(np.ceil(args.num_epoch * 0.75)),
                        int(np.ceil(args.num_epoch * 0.876)),
                    ]
                ),
            },
            "tiny_overfit": tiny_result,
        }
        metadata_path = os.path.join(run_dir, "preprocessing_and_run_config.json")
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(run_metadata, stream, indent=2, sort_keys=True)
        print("Run metadata       :", metadata_path)

        section("TRAIN FULL 2,177-SAMPLE BASELINE-STEP")
        train_loader = create_data_loader(
            train_data, train_labels, args.batch_size, True, args.num_worker
        )
        val_loader = create_data_loader(
            val_data, val_labels, args.batch_size, False, args.num_worker
        )
        test_loader = create_data_loader(
            test_data, test_labels, args.batch_size, False, args.num_worker
        )
        first_input, first_label = next(iter(train_loader))
        print("ST-GCN batch shape :", tuple(first_input.shape), "(N,C,T,V,M)")
        print("First label         :", int(first_label[0]), metadata["emotion_names"][int(first_label[0])])
        print("Training work dir   :", full_dir)

        args.work_dir = full_dir
        full_processor = processor.Processor(
            args,
            {"train": train_loader, "val": val_loader, "test": test_loader},
            in_channels=3,
            num_classes=4,
            graph_dict={"strategy": "spatial"},
            device=device,
            verbose=True,
        )
        metrics = full_processor.train()
        run_metadata["results"] = {
            "best_epoch": metrics["best_epoch"],
            "train_accuracy": metrics["train_accuracy"],
            "validation_accuracy": metrics["best_val_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "test_loss": metrics["test_loss"],
        }
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(run_metadata, stream, indent=2, sort_keys=True)

        section("FULL BASELINE RESULT")
        print("Best epoch          :", metrics["best_epoch"])
        print("Train Top1          : {:.2f}%".format(metrics["train_accuracy"]))
        print("Best validation Top1: {:.2f}%".format(metrics["best_val_accuracy"]))
        print("Final test Top1     : {:.2f}%".format(metrics["test_accuracy"]))
        print("Final test loss     : {:.6f}".format(metrics["test_loss"]))
        print("Tiny checkpoint     :", tiny_result["checkpoint"])
        print("Run metadata        :", metadata_path)
        print("Log saved to        :", log_path)

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
