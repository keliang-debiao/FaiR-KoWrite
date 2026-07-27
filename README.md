# FaiR-KoWrite

This repository contains the anonymous review code package for **FaiR-KoWrite**, a fairness-aware framework for ordinal Korean second-language writing proficiency assessment. The package provides the core model implementation, data reconstruction and auditing utilities, training objectives, evaluation tools, experiment configurations, protocol manifests, validation schemas, test cases, and repository-integrity utilities.

This README describes only the contents of the package. It does not report experimental outcomes.

## Repository Structure

```text
FaiR-KoWrite_Anonymous_Review_Package/
├── fairkowrite/          # Core Python package
├── scripts/              # Data, training, evaluation, and audit entry points
├── configs/              # Main, baseline, ablation, robustness, and statistics settings
├── manifests/            # Data-source, split, provenance, revision, and audit metadata
├── schemas/              # JSON schemas for essays, external data, and predictions
├── data/                 # Data-placement and licensing instructions
├── environment/          # Conda and pip dependency specifications
├── tests/                # Unit and integrity tests
├── docs/                 # Anonymous-review and full-run integration documentation
├── logs/                 # Marked simulated logs and synthetic smoke-test logs
├── outputs/              # Output directory placeholders and smoke-test artifacts
└── results_reference/    # Machine-readable reference artifacts retained with the package
```

## Core Python Package

### `fairkowrite/data/`

Data processing and split-management utilities:

- `conllu.py` parses Korean learner-language CoNLL-U files, normalizes text, reconstructs sentence-level records into essay-level documents, assigns proficiency labels for the pinned release, and applies exact-text duplicate handling.
- `duplicates.py` implements character-shingle generation, MinHash signatures, estimated Jaccard similarity, exact Jaccard similarity, and near-duplicate candidate screening.
- `folds.py` loads writer-grouped fold manifests, attaches fold assignments to essays, and checks that writers do not cross fold boundaries.
- `form_vocab.py` provides the Jamo-and-byte form tokenizer used by the form-processing branch.

### `fairkowrite/models/`

Model components and baseline definitions:

- `semantic.py` contains a compact semantic encoder for smoke testing and a Hugging Face semantic-encoder wrapper for full-model integration.
- `form_encoder.py` implements the form branch with token embeddings, sequence-mixing layers, attention pooling, projection, normalization, and dropout.
- `mamba.py` provides a sequence-mixing wrapper that uses Mamba-2 when available and a deterministic GRU fallback for CPU and smoke validation.
- `fusion.py` implements cross-gated bilinear fusion between semantic and form representations.
- `discourse.py` implements bidirectional document-level discourse encoding and attention-based essay pooling.
- `corn.py` implements the conditional ordinal regression head, ordinal probability conversion, and CORN loss.
- `adversarial.py` contains the gradient-reversal operation and adversarial discriminators.
- `groupdro.py` implements level-aware Group Distributionally Robust Optimization weighting.
- `model.py` assembles the semantic encoder, form encoder, cross-gated fusion module, discourse encoder, ordinal head, source discriminator, and first-language discriminator into the FaiR-KoWrite model.
- `baselines.py` contains a character TF-IDF ordinal baseline, a bidirectional LSTM attention baseline, and a hierarchical attention baseline. Additional pretrained baseline definitions are listed in the configuration files.

### `fairkowrite/training/`

Training logic:

- `losses.py` combines the ordinal GroupDRO objective with optional source-adversarial, first-language-adversarial, and consistency losses.
- `trainer.py` provides an AdamW-based training step with gradient clipping and GroupDRO weight updates.

### `fairkowrite/evaluation/`

Evaluation and analysis utilities:

- `metrics.py` provides ordinal classification and error metrics, expected-score conversion, and expected calibration error.
- `calibration.py` implements temperature scaling, calibrated probability conversion, ensemble uncertainty estimation, and selective prediction metrics.
- `shift.py` provides representation-shift analysis through RBF maximum mean discrepancy, writer-held-out probes, and out-of-distribution scoring metrics.
- `robustness.py` implements normalization, punctuation, spacing, Jamo substitution, sentence-order, and truncation transformations.
- `statistics.py` contains writer-clustered bootstrap intervals, paired writer-level permutation tests, and Holm correction.
- `efficiency.py` provides parameter counting and inference-time benchmarking helpers.

### `fairkowrite/utils/`

General utilities:

- `io.py` contains JSONL, JSON, and YAML input/output helpers.
- `hashing.py` contains SHA-256 file and text hashing functions.
- `seed.py` provides deterministic random-seed initialization.

## Scripts

The `scripts/` directory contains the package entry points:

| Script | Included function |
|---|---|
| `00_download_and_verify.py` | Downloads the pinned public UD Korean-KSL files and verifies their SHA-256 hashes. |
| `01_reconstruct_audit.py` | Reconstructs KH and ARG essays, applies exact duplicate handling, and writes reconstruction audit files. |
| `01b_near_duplicate_audit.py` | Screens KH, ARG, and optional KoLLA records for near-duplicate candidates. |
| `02_build_folds.py` | Creates writer-grouped, proficiency-stratified outer-fold manifests. |
| `03_train_cv.py` | Creates the full cross-validation run plan and enforces the licensed-data and immutable-model-revision boundary. |
| `04_ensemble_oof.py` | Averages fold-and-seed prediction files into an out-of-fold prediction table. |
| `05_calibrate_abstain.py` | Fits temperature scaling from cross-fitted validation logits and prepares selective-prediction outputs. |
| `06_arg_diagnostics.py` | Runs first-language probing and representation-shift diagnostics on fold-specific ARG representations. |
| `07_leave_one_l1_out.py` | Generates leave-one-first-language-out retraining plans. |
| `08_kolla_external.py` | Reads frozen external KoLLA predictions and evaluates their association with rubric totals. |
| `09_robustness.py` | Applies the supported text-robustness transformations. |
| `10_strict_prompt_writer.py` | Defines the strict prompt-and-writer transfer split and retraining protocol. |
| `11_statistics.py` | Runs writer-clustered bootstrap and optional paired permutation analyses. |
| `12_efficiency.py` | Reports model parameter counts and points to the efficiency reference artifact. |
| `13_run_ablations.py` | Generates fold-and-seed run plans for every configured ablation variant. |
| `14_train_baselines.py` | Runs the transparent TF-IDF ordinal baseline or prepares the shared protocol for neural baselines. |
| `run_smoke.py` | Performs a lightweight end-to-end smoke test using synthetic, non-learner tensors. |
| `_synthetic.py` | Creates deterministic synthetic batches for smoke testing. |
| `verify_repository.py` | Checks required repository files, anonymous-review constraints, marked simulated artifacts, and strict-submission conditions. |
| `generate_simulated_logs.py` | Generates explicitly marked simulated reference logs for workflow inspection. |
| `create_checksums.py` | Creates package-level SHA-256 checksums. |
| `check_checksums.py` | Verifies package files against the stored checksum list. |

## Configuration Files

### `configs/main.yaml`

Contains the shared project protocol, dataset paths, random seeds, model dimensions, semantic-encoder settings, form-encoder settings, Mamba settings, LoRA settings, cross-validation settings, optimization parameters, loss weights, GroupDRO settings, gradient-reversal settings, calibration settings, and statistical-analysis settings.

### `configs/baselines/models.yaml`

Lists the configured comparison models, including the character TF-IDF ordinal model, recurrent and hierarchical attention models, multilingual transformer encoders, the Granite embedding encoder, and the Qwen embedding baseline.

### `configs/ablations/variants.yaml`

Defines the full model and ablation variants for removing or replacing the form branch, semantic branch, fusion mechanism, discourse module, ordinal head, adversarial objectives, GroupDRO, consistency objective, and calibration stage.

### `configs/robustness.yaml`

Defines equivalent-form and substantive text perturbations, their severity settings, and the robustness random seed.

### `configs/statistics.yaml`

Defines writer-clustered resampling settings, paired-comparison settings, multiple-comparison families, external-validation resampling settings, and the statistics random seed.

## Manifests

The `manifests/` directory contains machine-readable metadata used to preserve the evaluation protocol:

- `data_sources.json`: public data-source identifiers, retrieval metadata, licenses, file hashes, and the external-data role.
- `model_revisions.json`: model identifiers, revision fields, licenses, and model roles.
- `writer_grouped_outer_folds.csv`: writer-level outer-fold assignments.
- `essay_outer_folds.csv`: essay-level fold assignments.
- `duplicate_decisions.csv`: retained and removed duplicate identifiers with decision reasons.
- `generated_fold_profile.csv`: profile of generated fold assignments.
- `paper_outer_fold_profile.csv`: experiment-locked fold-profile reference.
- `fold_manifest_provenance.json`: fold-manifest provenance metadata.
- `reconstruction_expected.json`: expected reconstruction-audit structure.
- `verified_data_hash_report.json`: stored data-hash verification metadata.

## Schemas

The `schemas/` directory defines the expected structure of data and prediction records:

- `essay.schema.json`: KH and ARG essay records.
- `kolla.schema.json`: independently obtained KoLLA external-validation records.
- `prediction.schema.json`: essay-level fold-and-seed ordinal probability records.

## Data Directory

The repository does not redistribute raw learner text. `data/README.md` describes:

- placement of the pinned public UD Korean-KSL files;
- placement of independently obtained KoLLA records;
- handling of derived text-bearing files;
- use of anonymized identifiers, manifests, hashes, schemas, and prediction files for anonymous review.

## Environment Specifications

The `environment/` directory contains:

- `environment.yml`: the Conda environment definition;
- `requirements-core-lock.txt`: the core Python, numerical, machine-learning, and testing dependencies;
- `requirements-full-reference.txt`: the extended transformer, parameter-efficient fine-tuning, tokenizer, Mamba, and model-serialization dependencies.

## Tests

The `tests/` directory contains checks for:

- form-vocabulary tokenization;
- ordinal probability and loss behavior;
- model forward propagation;
- writer-grouped fold handling;
- mandatory markers on simulated artifacts.

## Documentation

The `docs/` directory contains:

- `ANONYMOUS_REPOSITORY_CHECKLIST.md`: repository-anonymity, split-integrity, calibration, external-validation, artifact-labeling, and checksum checks.
- `FULL_RUN_INTEGRATION.md`: the boundary between the supplied scientific implementation and institution-specific licensed-data, model-snapshot, storage, scheduler, and compute integration.

## Logs and Output Artifacts

- `logs/simulated/` contains reference logs that are explicitly marked as simulated and not measured outputs.
- `logs/smoke/` contains logs from the synthetic, non-learner smoke-validation workflow.
- `outputs/` is the target directory for generated audit, prediction, calibration, diagnostic, ablation, baseline, and efficiency artifacts.
- `outputs/smoke/` contains the bundled smoke-test report.

## Reference Artifacts

The `results_reference/` directory contains machine-readable files retained for package-level comparison and workflow checking:

- `paper_targets.json`;
- `main_results.csv`;
- `computational_efficiency.csv`;
- `near_duplicate_audit.csv`;
- a directory-level `README.md` describing the status of these files.

No numerical values from these artifacts are reproduced in this README.
