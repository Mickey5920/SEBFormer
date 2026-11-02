# SEBFormer

Minimal PyTorch implementation of SEBFormer with baseline and event preprocessing.
The model retains the source-specific encoders, shared bottleneck fusion and
linear prediction head used in the revised manuscript.

## Run

```bash
python -m pip install -r requirements.txt
python demo.py
```

The CPU example uses synthetic data and random weights. Expected output:

```text
Synthetic preprocessing and model forward: OK
Main logits: (2, 1507)
Auxiliary logits: (2, 5, 1507)
```

## Files

- `model.py`: complete model in one file.
- `preprocess.py`: training-only imputation, clipping, standardization, missingness
  indicators, categorical encoding, event vocabulary and sequence padding.
- `demo.py`: preprocessing followed by a model forward pass.

## Your data

Provide nine DataFrames keyed by `demo`, `physical`, `biomarkers`, `lifestyle`,
`mental`, `repro`, `environmental`, `genetic` and `other`. Their indices must identify
the same rows in the same order. Use feature columns only; `demo` must contain
`sex` and `age`, and `physical` must contain `bmi`. Represent unavailable numeric
values as NaN after handling dataset-specific missing-value codes.

Fit `BaselinePreprocessor` on training rows, then call `transform` on each split.
It returns nine float32 arrays; `group_dims` supplies their widths for the model.
Use `save`/`load` to reuse the fitted preprocessing.

Supply each coded source as a list of participants' `[(code, years_from_baseline),
...]` records. Use date/code order for histories and field-name order for observed
baseline phenotype fields; preserve repeated observations. Times are nonpositive
years, with zero for baseline phenotype fields. Imputed values do not create
phenotype tokens. `build_vocab` creates a shared vocabulary; reuse it unchanged.
`encode_events` keeps the first 12 and last 116 entries when longer than 128,
uses 0 for padding and 1 for unknown codes, and returns `[batch, 128, 2]` arrays.

Pass the nine baseline tensors, then diagnoses, phenotype fields, procedures and
prescriptions to `model(*inputs)`. It returns main logits and auxiliary logits;
`torch.sigmoid(logits)` gives the main scores. For learning, the provided
`effective_binary_mask(labels, at_risk, observed)` combines already-defined
endpoint eligibility and observation masks.

This package covers model-input preprocessing, not cohort extraction from raw
UKB/MIMIC/CHARLS exports. Data, fitted preprocessing and trained weights are not
included.
