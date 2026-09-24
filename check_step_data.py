# check_step_data.py

import os
import h5py
import numpy as np
from collections import Counter


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FILES = [
    "features.h5",
    "labels.h5",
    "features_ELMD.h5",
    "labels_ELMD.h5",
    "affectiveFeatures.h5",
    "affectiveFeatures_ELMD.h5",
]


def summarize_h5(path):
    print("\n" + "=" * 80)
    print(os.path.basename(path))
    print("=" * 80)

    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        print("num keys:", len(keys))
        print("first 10 keys:")
        for k in keys[:10]:
            print(" ", k)

        if not keys:
            return

        shapes = []
        dtypes = []
        finite_flags = []
        mins = []
        maxs = []

        for k in keys:
            arr = np.asarray(f[k])

            shapes.append(arr.shape)
            dtypes.append(str(arr.dtype))

            if np.issubdtype(arr.dtype, np.number):
                finite_flags.append(np.isfinite(arr).all())

                if arr.size > 0:
                    mins.append(np.nanmin(arr))
                    maxs.append(np.nanmax(arr))

        print("\nshape distribution:")
        for shape, count in Counter(shapes).most_common(20):
            print(" ", shape, "->", count)

        print("\ndtype distribution:")
        for dtype, count in Counter(dtypes).most_common():
            print(" ", dtype, "->", count)

        if finite_flags:
            print("\nall numeric arrays finite:", all(finite_flags))

        if mins:
            print("global min:", min(mins))
            print("global max:", max(maxs))

        print("\nexample samples:")
        for k in keys[:5]:
            arr = np.asarray(f[k])
            print(
                " key =", k,
                "| shape =", arr.shape,
                "| dtype =", arr.dtype
            )


def inspect_labels(path):
    print("\n" + "=" * 80)
    print("LABEL INSPECTION:", os.path.basename(path))
    print("=" * 80)

    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        print("num keys:", len(keys))
        print("first 10 keys:", keys[:10])

        all_values = []

        for k in keys:
            arr = np.asarray(f[k])

            flat = arr.reshape(-1)

            for x in flat:
                try:
                    all_values.append(int(x))
                except Exception:
                    pass

        if all_values:
            counts = Counter(all_values)

            print("label distribution:")
            for label, count in sorted(counts.items()):
                print(" ", label, "->", count)


def compare_key_alignment(path_a, path_b, name_a, name_b):
    print("\n" + "=" * 80)
    print("KEY ALIGNMENT")
    print(name_a, "<->", name_b)
    print("=" * 80)

    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        ka = list(fa.keys())
        kb = list(fb.keys())

        set_a = set(ka)
        set_b = set(kb)

        print(name_a, "keys:", len(ka))
        print(name_b, "keys:", len(kb))

        print("same exact key set:", set_a == set_b)
        print("intersection:", len(set_a & set_b))
        print("only in", name_a, ":", len(set_a - set_b))
        print("only in", name_b, ":", len(set_b - set_a))

        if set_a != set_b:
            print("\nexamples only in", name_a)
            for x in list(set_a - set_b)[:10]:
                print(" ", x)

            print("\nexamples only in", name_b)
            for x in list(set_b - set_a)[:10]:
                print(" ", x)


def inspect_skeleton_file(path, name):
    print("\n" + "=" * 80)
    print("SKELETON CHECK:", name)
    print("=" * 80)

    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        lengths = []
        widths = []

        for k in keys:
            arr = np.asarray(f[k])

            if arr.ndim != 2:
                print("WARNING non-2D sample:", k, arr.shape)
                continue

            lengths.append(arr.shape[0])
            widths.append(arr.shape[1])

        print("samples:", len(keys))

        if lengths:
            print("timesteps min:", min(lengths))
            print("timesteps max:", max(lengths))
            print("timesteps mean:", np.mean(lengths))

        if widths:
            print("feature widths:", sorted(set(widths)))

            if len(set(widths)) == 1:
                w = widths[0]

                if w % 3 == 0:
                    print("implied joints assuming xyz:", w // 3)


def main():
    print("DATA DIR:", DATA_DIR)

    for filename in FILES:
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            print("\nMISSING:", path)
            continue

        summarize_h5(path)

        if "labels" in filename.lower():
            inspect_labels(path)

    features = os.path.join(DATA_DIR, "features.h5")
    labels = os.path.join(DATA_DIR, "labels.h5")

    features_elmd = os.path.join(DATA_DIR, "features_ELMD.h5")
    labels_elmd = os.path.join(DATA_DIR, "labels_ELMD.h5")

    aff = os.path.join(DATA_DIR, "affectiveFeatures.h5")
    aff_elmd = os.path.join(DATA_DIR, "affectiveFeatures_ELMD.h5")

    if os.path.exists(features):
        inspect_skeleton_file(features, "features.h5")

    if os.path.exists(features_elmd):
        inspect_skeleton_file(features_elmd, "features_ELMD.h5")

    if os.path.exists(features) and os.path.exists(labels):
        compare_key_alignment(
            features,
            labels,
            "features.h5",
            "labels.h5",
        )

    if os.path.exists(features_elmd) and os.path.exists(labels_elmd):
        compare_key_alignment(
            features_elmd,
            labels_elmd,
            "features_ELMD.h5",
            "labels_ELMD.h5",
        )

    if os.path.exists(features) and os.path.exists(aff):
        compare_key_alignment(
            features,
            aff,
            "features.h5",
            "affectiveFeatures.h5",
        )

    if os.path.exists(features_elmd) and os.path.exists(aff_elmd):
        compare_key_alignment(
            features_elmd,
            aff_elmd,
            "features_ELMD.h5",
            "affectiveFeatures_ELMD.h5",
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()