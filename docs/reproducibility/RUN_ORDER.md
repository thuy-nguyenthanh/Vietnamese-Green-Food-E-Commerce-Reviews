# Reproducibility and Release Verification

Run commands from the repository root after installing `requirements.txt`.

The package is organized around the paper results. It includes processed data and archived outputs for public audit. The original raw-crawl workspace is not redistributed, so some historical pipeline scripts are provided as methodological code snapshots rather than a one-command raw-to-paper workflow.

## Required Public-Release Check

Before uploading or tagging a release:

```bash
python code/verify_release.py
```

This check verifies row counts, scans released text fields for direct identifiers, flags absolute local paths, and validates JSON files.

## Auditable Outputs

Use the included archived outputs for manuscript numbers:

- Dataset funnel and corpus profile: `data/publication/phase55_final_001/`
- T2 aspect holdout evaluation: `outputs/evaluation/aspect_t2_gold_holdout/`
- T3 ViSoBERT sentiment diagnosis: `outputs/evaluation/sentiment_t3_primary_no_adrd25/`
- T3 sensitivity and error analysis: `outputs/evaluation/sentiment_t3_sensitivity/`
- Post-hoc SHAP pilot: `outputs/evaluation/shap_pilot_phase55_final_001/`

## LLM-Assisted Annotation Prompt

The Step F prompt used to generate silver extended labels is archived at
`docs/reproducibility/LLM_ANNOTATION_PROMPT.md`. It documents the annotation
schema and few-shot examples for the LLM-assisted pass only; the resulting
silver labels are not human-gold evaluation labels.

## Optional Script Review

The scripts under `code/` document the analysis logic used in the research workspace. If you want to re-run them from this public package, first inspect input paths and adapt them to the released `data/` and `outputs/` layout.

## Notes

- T2/T3 results are based on human-gold evaluation files.
- Silver labels are used for descriptive analysis only.
- BERTopic and SHAP outputs are exploratory.
- SHAP values are feature attributions, not causal effects.
