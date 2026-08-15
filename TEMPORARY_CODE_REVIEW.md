# scdiag Temporary Code Review

**Review scope:** repository snapshot at commit `76220c5` (working tree was clean before this review), including `README.md`, `pyproject.toml`, the `scdiag/` package, `scripts/`, and `tests/`.

**Review date:** Current working-session review

**Purpose:** This is an actionable engineering review, not a claim that every item is a confirmed defect. Items marked **observed** are directly supported by the current code or verification commands; items marked **risk/opportunity** need a product or deployment decision before implementation.

## Executive summary

The project has a useful, well-tested core: the existing test suite passes **335 tests**, checkpointing is shared between training and pre-training, the public README is substantial, and the code has clear extension points for models, classifiers, optimizers, schedulers, and datasets. The highest-priority engineering concerns are:

1. **Dataset failures can be silently replaced by random samples**, changing the effective training distribution and hiding corrupt data.
2. **Automatic label/image detection and label normalization can create silent dataset/schema errors**, especially when IDs, strings, or multiple candidate columns are present.
3. **Gradient accumulation and epoch accounting need careful validation**: the final partial accumulation window is stepped as a full window, while training uses `drop_last=True`, and scheduler semantics are not recorded or validated.
4. **Checkpoint resume is permissive but does not validate the experiment contract**: architecture, label map, preprocessing, dataset split, optimizer configuration, and loader length can change while a run appears to resume successfully.
5. **Reproducibility and experiment provenance are incomplete**, and the lint gate currently fails even though the test suite passes.

## Impact/complexity legend

- **Impact:** High (data integrity, reproducibility, or likely production failure), Medium (meaningful reliability/maintainability/performance effect), Low (localized polish or developer experience).
- **Complexity:** Low (hours to a day), Medium (a few days or a contained design change), High (cross-cutting or requiring compatibility/performance work).
- Priority is a recommendation, not an implementation schedule.

## Findings matrix

| ID | Area | Impact | Complexity | Priority |
|---|---|---:|---:|---:|
| D1 | Random retry masks corrupt samples and biases data | High | Low–Medium | P0 |
| D2 | Heuristic schema detection and label normalization | High | Medium | P0 |
| T3 | Gradient accumulation and dropped-sample accounting | High | Medium | P1 |
| M1 | Permissive checkpoint resume lacks experiment validation | High | Medium | P1 |
| R1 | Incomplete reproducibility and provenance capture | High | Medium–High | P1 |
| T1 | Lint gate currently fails despite tests passing | Medium | Low | P1 |
| T2 | Missing end-to-end and failure-path coverage | Medium | Medium | P1 |
| P1 | Data loading, caching, and image lifecycle controls | Medium | Medium | P2 |
| P2 | Metrics, class imbalance, and split methodology | Medium | Medium–High | P2 |
| E1 | CLI/schema and error-reporting improvements | Medium | Medium | P2 |
| C1 | Dependency and release hygiene | Medium | Medium | P2 |
| L1 | Documentation consistency and operational runbooks | Low–Medium | Low–Medium | P2 |
| A1 | API typing and package architecture | Low–Medium | Medium–High | P3 |

# High-impact data and correctness items

## D1 — Random retry fallback hides bad data and changes the effective dataset

**Evidence:** `scdiag/datasets/retry.py:30-41` catches `Exception` and, after failure, replaces the requested index with a random index. `datasets/ensemble.py` and image-loading paths use this mechanism.

**Risk:** A corrupt image, invalid label, transient storage error, or programming bug is silently converted into another sample. This can duplicate easy examples, distort class frequencies, make evaluation nondeterministic, and prevent operators from finding bad files. Catching every exception also hides errors such as index bugs.

**Recommendation:** Default to fail-fast or return a structured invalid-sample result with a bounded, deterministic retry policy. Catch only expected I/O/decode exceptions. Count failures by dataset/index/error type, enforce a maximum failure rate, and abort validation if the threshold is exceeded. For training, consider a preflight scan/quarantine mode rather than silently substituting samples.

## D2 — Automatic column detection and label normalization need strict validation

**Evidence:** `scdiag/datasets/hf_proxy.py:15-203` uses known names and feature types to infer image and label columns. The fallback accepts string or `int64` values and skips only a fixed ignore list. `normalize_labels()` calls `class_encode_column()` on the proxy’s dataset (`hf_proxy.py:148-160`), so independently wrapped splits can construct independent class-to-ID mappings when the source label is a plain string column.

**Risk:** On datasets with multiple image-like or scalar columns, the selected fields may be wrong while still producing valid tensors. More seriously, separately encoded train and validation splits can assign different integer IDs to the same label set (and validation may lack or reorder classes), producing incorrect metrics or out-of-range targets without an obvious schema error. Labels can be remapped unexpectedly, and filepath handling can fail late.

**Recommendation:** Resolve image/label columns once before splitting and create one shared, explicit label vocabulary/mapping for every split. Prefer preserving a source `ClassLabel`; otherwise encode the complete source dataset before splitting, or pass a shared mapping into each proxy. Add explicit `--image_column` and `--label_column` overrides and log the selected columns, feature types, class names, and mapping. In strict mode, fail when multiple candidates exist, when split mappings differ, when validation contains unknown labels, or when inferred cardinality/types are suspicious. Add a dataset inspection command and preflight checks for unreadable images, empty classes, duplicate IDs, and train/validation leakage.

## T3 — Gradient accumulation and dropped-sample accounting can change optimization

**Evidence:** `train.py:865-910` divides every micro-batch loss by `args.grad_accum_steps`, then steps on the final micro-batch group even when its size is not a multiple of that value. The training loader uses `drop_last=True` (`train.py:1109-1116`), but this only removes an incomplete *sample batch*, not an incomplete accumulation window. The scheduler is stepped once per epoch (`train.py:1257-1258`).

**Risk:** The final partial accumulation window contributes fewer gradients than intended because it is divided by the full accumulation count. Its relative weight depends on the number of batches per epoch. Reported loss is reconstructed with a compensating multiplication, so logs can look normal while the parameter update is biased. A scheduler configured for per-step semantics also silently receives epoch-level steps.

**Recommendation:** Track the actual number of micro-batches in each accumulation window and normalize by that count, or deliberately drop incomplete windows and report the number discarded. Make scheduler cadence explicit (`step` versus `epoch`) and validate scheduler arguments against the selected cadence. Add tests where the number of batches is not divisible by `grad_accum_steps`, including mixup and AMP.

## R1 — Reproducibility and provenance are not end-to-end

**Evidence:** `train.py` seeds dataset splitting (`train.py:221-269`), but the review found no single experiment manifest covering all CLI arguments, package versions, model/processor revisions, dataset revisions, random states, worker initialization, or checkpoint provenance. Data loaders use worker processes (`train.py:1109-1122`, `pretrain.py:537-543`).

**Risk:** Re-running a command may produce different samples, augmentations, initialization, or cached artifacts. This weakens scientific comparability and makes a regression difficult to reproduce.

**Recommendation:** Add one seed option and seed Python, NumPy, Torch CPU/CUDA, DataLoader workers, and all dataset shuffles. Provide deterministic mode with a clear performance tradeoff. Save a JSON manifest beside every checkpoint containing the resolved CLI configuration, git commit/diff, dependency versions, hardware, model/dataset identifiers, label map, preprocessing, seed, split indices or split hash, and scaler/scheduler settings. Record local artifact versions or hashes when inputs are cached.

## M1 — Checkpoint compatibility should be explicit and fail-safe

**Evidence:** `checkpointing.py:330-343` filters missing/shape-mismatched state entries, while `:510-520` logs mappings and missing keys. The flexible alignment behavior is useful but can allow a partially loaded model to continue.

**Risk:** A wrong architecture, label map, processor, or classifier head may load “successfully” with missing or skipped weights and produce misleading results. Resume state may also be incompatible with a changed optimizer/model configuration.

**Recommendation:** Put a schema version, model identifier/config hash, label-map hash, preprocessing hash, and tensor manifest in each checkpoint. Make strict compatibility the default for resume; require an explicit `--allow-partial-load` for transfer learning. Report skipped/missing keys in a machine-readable summary and enforce minimum match ratios. Validate optimizer/scheduler/scaler state against the selected resume policy.

# Medium-impact reliability and engineering items

## T1 — Ruff is not clean

**Observed verification:** `pytest -q` completed with **335 passed**. `ruff check scdiag tests` failed with one `SIM117` finding in `scdiag/models/cls_model_wrapper/model.py:97-100`, recommending a combined context manager:

```python
with model_mode(self, "eval"), torch.no_grad():
```

**Recommendation:** Fix the lint finding using the project’s required YAPF configuration, then make Ruff and tests required CI checks. Avoid adding suppression comments unless a specific exception is reviewed.

## T2 — Test coverage is broad but lacks safety and failure-path tests

The 335 tests cover many core components, but the review recommends adding tests for: independently encoded split mappings; unknown validation labels; corrupt image accounting; ambiguous dataset columns; empty/single-class datasets; invalid CLI values and malformed KV pairs; checkpoint schema/version mismatches; interrupted/atomic checkpoint writes; non-divisible gradient accumulation; scheduler cadence; CUDA/AMP branches; multiple DataLoader workers; and exact reproducibility across two runs.

Add a small end-to-end fixture that trains, resumes, evaluates, and infers using a local deterministic dataset, then validates the saved manifest and predictions.

## P1 — Data-loader controls and image lifecycle

`DataLoader` construction in `train.py`, `pretrain.py`, and `model_utils.py` has basic worker/pin-memory settings, but configurable `persistent_workers`, `prefetch_factor`, `timeout`, `pin_memory_device`, and worker seeding would improve throughput and diagnosis. PIL images should be closed or loaded into memory deliberately when opening paths/streams. Input files should be validated before `Image.open`, and a shared image-loading policy should define supported formats and maximum dimensions.

Pre-training uses `drop_last=True` (`pretrain.py:543`), which is often required for batch objectives but should be reported because it discards samples. Add loader statistics and make the tradeoff visible in the run manifest.

## P2 — Metrics, class imbalance, and split methodology require an explicit policy

Training reports precision/recall/F1, including macro F1, but the split path uses an unstratified `train_test_split` (`train.py:241-269`). A small or imbalanced medical dataset can therefore produce validation sets with missing classes; the current inverse-frequency weights also clamp absent classes to count one (`train.py:313-321`), which gives nonexistent training classes a finite weight rather than flagging a split/schema problem.

Document and, where appropriate, implement stratified splitting or group/patient-level splitting. Never split correlated images from the same patient across train and validation without an explicit group key. Report class counts for both splits and fail or warn when a class is absent. Add balanced accuracy, per-class support, confusion matrix, AUROC/AUPRC where valid, and confidence/calibration reporting. State whether class weights, weighted sampling, threshold tuning, calibration, and multilabel targets are supported.

## E1 — CLI schemas and diagnostics can be safer

`cli_utils.py:98-108` parses arbitrary `KEY=VALUE` tokens, which is convenient but weakly typed. Invalid values may reach model/optimizer constructors and fail late. Add typed schemas per component, validation of unknown keys, consistent boolean/list/tuple parsing, and a `--print-config`/resolved-config output. Avoid exposing secrets in logged kwargs or manifests. Replace terse fatal exits with errors that include the argument, accepted values, and remediation.

## C1 — Dependency and release hygiene

`pyproject.toml` declares broad/unbounded runtime dependencies (`numpy`, `torch`, `torchvision`, `transformers`, `datasets`) while the README lists requirements not all visibly constrained in the project metadata. This can produce incompatible installs and unreproducible environments. Define tested version ranges, separate CPU/CUDA installation guidance, and publish a lock/constraints file for CI and reference experiments. Add package metadata for supported Python versions, release notes, and a CI matrix.

Also consider lazy imports for optional integrations and verify that optional `gcs`, `lora`, and XGBoost paths fail with an actionable install message rather than an import traceback.

# Lower-impact maintainability and usability items

## L1 — Documentation should distinguish stable behavior from implementation details

The README is extensive and documents installation, quick start, CLI arguments, custom models/classifiers, and development. Improvements:

- Document the exact split semantics, stratification/grouping policy, random seed behavior, retry behavior, dropped batches, label mapping, gradient accumulation, scheduler cadence, and partial checkpoint loading.
- Add a supported-model/backend compatibility table and CPU-only smoke-test instructions.
- Add an example showing explicit image/label columns and a reproducible manifest.
- Document output JSON schema and version it if it is consumed downstream.
- Add troubleshooting for common HF cache/auth, CUDA/AMP, corrupt-image, and resume failures.
- Add a deprecation/versioning policy and mark `TEMPORARY_CODE_REVIEW.md` as an internal review artifact rather than user documentation.

## A1 — Type annotations and boundaries

Most modules use docstrings but have limited type annotations. Adding annotations for public functions (`load_model`, dataset adapters, checkpoint APIs, CLI parsers, metric outputs) would make contracts clearer and improve editor/static-checker support. Small typed dataclasses would be preferable to loosely shaped dictionaries for checkpoint metadata, model outputs, inference results, and resolved CLI configuration.

The registry (`scdiag/models/registry.py:18-20`) relies on mutable module-level dictionaries and import-time registration. This is simple, but plugin discovery, duplicate-registration diagnostics, and explicit backend interfaces would improve extensibility and test isolation.

# Suggested implementation sequence

1. **P0 data integrity:** establish one shared label mapping across splits; make dataset columns explicit/strict; replace random fallback with observable bounded handling; add preflight validation.
2. **P1 optimization/resume:** fix partial gradient accumulation normalization; define scheduler cadence; add checkpoint experiment-compatibility checks and atomic-write tests.
3. **P1 reproducibility/CI:** add manifests and complete seeding; fix Ruff; make pytest + Ruff required gates.
4. **P2 evaluation/operations:** implement documented split/grouping and class-balance policy, improve metrics, tune loaders, and add typed CLI validation.
5. **P3 maintainability:** add types, plugin boundaries, version constraints, and documentation/runbooks.

# Verification performed

- `pytest -q`: **335 passed**.
- `ruff check scdiag tests`: **failed** with one `SIM117` violation in `scdiag/models/cls_model_wrapper/model.py:97-100`.
- No files other than this review were intentionally modified; no Git commit was created.

# Limitations

This review is based on static inspection and the existing automated tests. It did not run a real HuggingFace dataset transfer, GCS transfer, CUDA/AMP job, multi-worker production workload, or large multi-source dataset. The label-mapping, accumulation, scheduler, split, and checkpoint-resume paths should receive focused integration tests after the highest-priority fixes.
