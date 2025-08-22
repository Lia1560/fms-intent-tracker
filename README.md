
# FMS Intent Tracker

This repo runs a weekly pipeline (via GitHub Actions) that collects and scores signals from FMS (Freelancer Management Systems) and the wider freelance economy.

## Outputs (every Monday 12:00 London)
- **Weekly Markdown brief** → `/reports/YYYY-MM-DD.md`
- **Longform blog post** → `/docs/_posts/YYYY-MM-DD-top-discoveries-longform.md`
- **CSV/JSON dumps** (for auditing) → `/reports/items-YYYY-MM-DD.csv/json`

The pipeline filters, clusters, and ranks news from vendors, analysts, regulators, and related sources using semantic scoring.

## Blog
Published automatically to GitHub Pages:  
👉 [https://lia1560.github.io/fms-intent-tracker](https://lia1560.github.io/fms-intent-tracker)

## Config
- `sources.yaml` – list of feeds and domains (starting with the big beefy set).  
- `config.yaml` – scoring weights and thresholds (tweak `keep_threshold` if results are too noisy or too empty).

## License
MIT License
