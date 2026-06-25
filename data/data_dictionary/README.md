# Data Dictionary

This folder should contain short field descriptions for the released CSV files.

Minimum fields to document:

- `review_id`: grouped review identifier used for leakage-controlled splitting.
- `clause_id`: clause-level unit identifier.
- `clause_text`: processed review clause text; direct identifiers and phone-like strings are masked where detected.
- `aspect_hint`: LLM-assisted silver aspect label.
- `sentiment_polarity_hint`: LLM-assisted silver sentiment label.
- `annotation_source`: provenance of the label or row.
- `split`: train, validation, or test assignment where available.

Do not include usernames, seller identifiers, or raw private platform metadata.
