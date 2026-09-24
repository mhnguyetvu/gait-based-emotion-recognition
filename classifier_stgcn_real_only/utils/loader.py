"""Input pipeline for the released, real-only E-Gait experiment."""

import os
import re

import h5py
import numpy as np
import torch


NUM_JOINTS = 16
NUM_COORDS = 3
NUM_CLASSES = 4
DEFAULT_TARGET_FRAMES = 75

# E-Gait README order. The classifier graph helpers use right-side branches
# before left-side branches, so every sample is explicitly converted to that
# single model order.
SOURCE_JOINT_ORDER = (
    "root", "spine", "neck", "head",
    "left_shoulder", "left_elbow", "left_hand",
    "right_shoulder", "right_elbow", "right_hand",
    "left_hip", "left_knee", "left_foot",
    "right_hip", "right_knee", "right_foot",
)
MODEL_JOINT_ORDER = (
    "root", "spine", "neck", "head",
    "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
    "right_hip", "right_knee", "right_foot",
    "left_hip", "left_knee", "left_foot",
)
JOINT_REORDER = tuple(SOURCE_JOINT_ORDER.index(name) for name in MODEL_JOINT_ORDER)

EMOTION_NAMES = {
    0: "Neutral",
    1: "Happy",
    2: "Angry",
    3: "Sad",
}


def read_and_verify_emotion_map(data_dir):
    """Check the on-disk E-Gait label map when its README is available."""
    map_path = os.path.join(data_dir, "annotation_N_emotions.md")
    if not os.path.isfile(map_path):
        return dict(EMOTION_NAMES), None

    found = {}
    in_four_emotion_section = False
    with open(map_path, "r", encoding="utf-8-sig") as stream:
        for line in stream:
            lowered = line.strip().lower()
            if lowered.startswith("annotation_4emotions.csv"):
                in_four_emotion_section = True
                continue
            if lowered.startswith("annotation_7emotions.csv"):
                in_four_emotion_section = False
            if not in_four_emotion_section:
                continue
            match = re.match(r"\s*(\d+)\s*:\s*([A-Za-z ]+)\s*$", line)
            if match:
                found[int(match.group(1))] = match.group(2).strip().title()

    normalized_expected = {key: value.title() for key, value in EMOTION_NAMES.items()}
    if found != normalized_expected:
        raise ValueError(
            "Unexpected E-Gait emotion mapping in {}. Expected {}, found {}."
            .format(map_path, normalized_expected, found)
        )
    return dict(EMOTION_NAMES), map_path


def temporal_resample(sequence, target_frames=DEFAULT_TARGET_FRAMES):
    """Linearly sample a (T, J, C) gait at target_frames evenly spaced times."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (NUM_JOINTS, NUM_COORDS):
        raise ValueError(
            "Expected (T, {}, {}) gait, received {}."
            .format(NUM_JOINTS, NUM_COORDS, sequence.shape)
        )
    if sequence.shape[0] < 2:
        raise ValueError("A gait must contain at least two frames.")
    if sequence.shape[0] == target_frames:
        return sequence.copy()

    source_positions = np.linspace(0.0, sequence.shape[0] - 1, target_frames)
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, sequence.shape[0] - 1)
    fraction = (source_positions - left).astype(np.float32)[:, None, None]
    return sequence[left] * (1.0 - fraction) + sequence[right] * fraction


def normalize_gait_geometry(sequence, target_frames=DEFAULT_TARGET_FRAMES):
    """Convert documented E-Gait joint order, root-center, then resample T."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 2 and sequence.shape[1] == NUM_JOINTS * NUM_COORDS:
        sequence = sequence.reshape(sequence.shape[0], NUM_JOINTS, NUM_COORDS)
    if sequence.ndim != 3 or sequence.shape[1:] != (NUM_JOINTS, NUM_COORDS):
        raise ValueError(
            "Expected E-Gait feature width 48 ((T,16,3)); received {}."
            .format(sequence.shape)
        )
    if not np.isfinite(sequence).all():
        raise ValueError("Gait contains NaN or Inf coordinates.")

    # Joint reordering is explicit, even though the two arm/leg branches are
    # topologically symmetric in this graph.
    sequence = sequence[:, JOINT_REORDER, :]

    # Make each frame relative to its root joint. This removes global
    # translation while retaining within-frame pose and across-frame motion.
    sequence = sequence - sequence[:, 0:1, :]
    sequence = temporal_resample(sequence, target_frames)
    if not np.isfinite(sequence).all():
        raise ValueError("Normalized gait contains NaN or Inf coordinates.")
    return sequence.astype(np.float32, copy=False)


def _read_scalar_label(dataset, key):
    raw_label = np.asarray(dataset[()])
    if raw_label.size != 1:
        raise ValueError(
            "Emotion label for {!r} must be scalar, got shape {}."
            .format(key, raw_label.shape)
        )
    value = float(raw_label.reshape(-1)[0])
    if not np.isfinite(value) or value != int(value):
        raise ValueError("Emotion label for {!r} is not a finite integer.".format(key))
    label = int(value)
    if label not in EMOTION_NAMES:
        raise ValueError(
            "Emotion label for {!r} is {}, expected one of {}."
            .format(key, label, sorted(EMOTION_NAMES))
        )
    return label


def load_merged_egait(
    data_dir,
    feature_filename="features_merged.h5",
    label_filename="labels_merged.h5",
    target_frames=DEFAULT_TARGET_FRAMES,
    expected_samples=2177,
):
    """Load, validate, reorder, root-center, and temporally resample E-Gait."""
    feature_path = os.path.join(data_dir, feature_filename)
    label_path = os.path.join(data_dir, label_filename)
    if not os.path.isfile(feature_path):
        raise FileNotFoundError("E-Gait feature file not found: {}".format(feature_path))
    if not os.path.isfile(label_path):
        raise FileNotFoundError("E-Gait label file not found: {}".format(label_path))

    emotion_names, emotion_map_path = read_and_verify_emotion_map(data_dir)
    with h5py.File(feature_path, "r") as feature_file, h5py.File(label_path, "r") as label_file:
        feature_keys = set(feature_file.keys())
        label_keys = set(label_file.keys())
        if feature_keys != label_keys:
            raise ValueError(
                "Feature/label HDF5 keys do not align: {} feature-only, {} label-only."
                .format(len(feature_keys - label_keys), len(label_keys - feature_keys))
            )

        keys = sorted(feature_keys)
        if expected_samples is not None and len(keys) != expected_samples:
            raise ValueError(
                "Expected {} merged real gaits, found {} in {}."
                .format(expected_samples, len(keys), feature_path)
            )
        if not keys:
            raise ValueError("Merged feature and label files contain no samples.")

        samples = []
        labels = []
        source_lengths = []
        for key in keys:
            feature = feature_file[key]
            label_data = label_file[key]
            if not isinstance(feature, h5py.Dataset):
                raise ValueError("Feature {!r} is not a dataset.".format(key))
            if not isinstance(label_data, h5py.Dataset):
                raise ValueError("Label {!r} is not a dataset.".format(key))

            raw = np.asarray(feature[()], dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != NUM_JOINTS * NUM_COORDS:
                raise ValueError(
                    "Feature {!r} must have shape (T,48), got {}."
                    .format(key, raw.shape)
                )
            source_lengths.append(int(raw.shape[0]))
            samples.append(normalize_gait_geometry(raw, target_frames))
            labels.append(_read_scalar_label(label_data, key))

    data = np.stack(samples).astype(np.float32, copy=False)
    labels = np.asarray(labels, dtype=np.int64)
    if data.shape != (len(keys), target_frames, NUM_JOINTS, NUM_COORDS):
        raise AssertionError("Unexpected preprocessed E-Gait shape: {}".format(data.shape))
    if not np.isfinite(data).all():
        raise ValueError("Preprocessed feature array contains NaN or Inf.")
    if set(np.unique(labels).tolist()) != set(EMOTION_NAMES):
        raise ValueError(
            "Expected all four emotion labels {}, found {}."
            .format(sorted(EMOTION_NAMES), np.unique(labels).tolist())
        )

    metadata = {
        "feature_path": feature_path,
        "label_path": label_path,
        "emotion_map_path": emotion_map_path,
        "emotion_names": emotion_names,
        "keys": keys,
        "source_lengths": source_lengths,
        "source_joint_order": list(SOURCE_JOINT_ORDER),
        "model_joint_order": list(MODEL_JOINT_ORDER),
        "joint_reorder_indices": list(JOINT_REORDER),
        "target_frames": int(target_frames),
    }
    return data, labels, metadata


class CoordinateNormalizer(object):
    """One train-fitted scalar RMS scale shared by all xyz coordinates."""

    def __init__(self):
        self.scale = None

    def fit(self, train_data):
        train_data = np.asarray(train_data, dtype=np.float64)
        if train_data.ndim != 4 or train_data.shape[2:] != (NUM_JOINTS, NUM_COORDS):
            raise ValueError("Normalizer expects (N,T,16,3), got {}.".format(train_data.shape))
        if not np.isfinite(train_data).all():
            raise ValueError("Cannot fit coordinate scale to NaN/Inf values.")

        # Exclude the root, which is identically zero after root-centering.
        rms = float(np.sqrt(np.mean(np.square(train_data[:, :, 1:, :]))))
        if not np.isfinite(rms) or rms < 1e-8:
            raise ValueError("Training coordinate RMS is invalid: {}.".format(rms))
        self.scale = rms
        return self

    def transform(self, data):
        if self.scale is None:
            raise RuntimeError("CoordinateNormalizer.fit() must be called first.")
        data = np.asarray(data, dtype=np.float32)
        normalized = data / np.float32(self.scale)
        if not np.isfinite(normalized).all():
            raise ValueError("Coordinate normalization produced NaN/Inf values.")
        return normalized.astype(np.float32, copy=False)


def stratified_split_7_2_1(labels, seed=42):
    """Return deterministic, class-stratified train/validation/test indices."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        count = len(class_indices)
        n_train = int(np.floor(0.70 * count))
        n_val = int(np.floor(0.20 * count))
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])

    shuffled = []
    for indices in (train_indices, val_indices, test_indices):
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        shuffled.append(indices)
    return tuple(shuffled)


def make_tiny_balanced_indices(labels, samples_per_class=4, seed=42):
    """Select a reproducible 4-per-class (16 total) memorization set."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in sorted(EMOTION_NAMES):
        indices = np.flatnonzero(labels == class_id).copy()
        if len(indices) < samples_per_class:
            raise ValueError(
                "Class {} ({}) has {} samples; need at least {}."
                .format(class_id, EMOTION_NAMES[class_id], len(indices), samples_per_class)
            )
        rng.shuffle(indices)
        selected.extend(indices[:samples_per_class])
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


class TrainTestLoader(torch.utils.data.Dataset):
    """Expose (N,T,J,C) arrays as ST-GCN (C,T,V,M) tensors."""

    def __init__(self, data, labels):
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        expected_tail = (DEFAULT_TARGET_FRAMES, NUM_JOINTS, NUM_COORDS)
        if data.ndim != 4 or data.shape[1:] != expected_tail:
            raise ValueError("Expected data shape (N,75,16,3), got {}.".format(data.shape))
        if labels.shape != (data.shape[0],):
            raise ValueError("Label shape {} does not match N={}.".format(labels.shape, data.shape[0]))
        if not np.isfinite(data).all():
            raise ValueError("Dataset contains NaN/Inf features.")
        self.data = np.transpose(data, (0, 3, 1, 2))[:, :, :, :, None].copy()
        self.labels = labels.copy()

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
