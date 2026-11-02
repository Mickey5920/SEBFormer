"""Run preprocessing and the full model on small synthetic inputs, using CPU."""

import numpy as np
import pandas as pd
import torch

from model import SEBFormer
from preprocess import GROUPS, BaselinePreprocessor, build_vocab, encode_events


def synthetic_frames():
    frames = {
        g: pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]}) for g in GROUPS
    }
    frames["demo"] = pd.DataFrame(
        {
            "sex": ["F", "M", "F", "M", "F", "M"],
            "age": [45.0, 52.0, 63.0, 58.0, 47.0, 69.0],
            "region": ["A", "B", "A", "B", "A", "C"],
        }
    )
    frames["physical"] = pd.DataFrame(
        {
            "bmi": [22.0, 25.0, np.nan, 28.0, 23.0, 26.0],
            "systolic": [120.0, 135.0, 140.0, np.nan, 125.0, 130.0],
        }
    )
    return frames


def main():
    torch.set_num_threads(4)
    torch.manual_seed(43)
    torch.set_float32_matmul_precision("highest")
    frames = synthetic_frames()
    train = {g: frame.iloc[:4] for g, frame in frames.items()}
    held_out = {g: frame.iloc[4:] for g, frame in frames.items()}
    preprocessor = BaselinePreprocessor().fit(train)
    baseline = preprocessor.transform(held_out)

    # Source order: diagnoses, observed phenotype fields, procedures, prescriptions.
    training_events = [
        [[("I10", -2.0), ("E11", -1.0)], [], [], []],
        [[("age", 0.0), ("bmi", 0.0)], [("age", 0.0)], [], []],
        [[("procedure_A", -3.0)], [], [], []],
        [[("drug_A", -1.0)], [], [], []],
    ]
    vocab = build_vocab(*training_events)
    held_out_events = [
        [[("I10", -1.0), ("unseen_diagnosis", -0.5)], []],
        [[("age", 0.0), ("bmi", 0.0)], []],
        [[("procedure_A", -2.0)], []],
        [[("drug_A", -1.0)], []],
    ]
    event_arrays = [encode_events(stream, vocab) for stream in held_out_events]
    inputs = [torch.from_numpy(x) for x in baseline + event_arrays]
    model = SEBFormer(
        preprocessor.group_dims, num_labels=1507, vocab_size=len(vocab), seed=43
    ).eval()
    with torch.inference_mode():
        logits, extra = model(*inputs)
    assert logits.shape == (2, 1507) and torch.isfinite(logits).all()
    assert extra["branch_logits"].shape == (2, 5, 1507)
    assert torch.isfinite(extra["branch_logits"]).all()
    print("Synthetic preprocessing and model forward: OK")
    print("Main logits:", tuple(logits.shape))
    print("Auxiliary logits:", tuple(extra["branch_logits"].shape))


if __name__ == "__main__":
    main()
