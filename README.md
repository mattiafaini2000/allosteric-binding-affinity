# Allosteric Binding Affinity: Data Curation and Baseline Diagnostics

This repository studies how activity and affinity labels can be linked to allosteric protein–ligand complexes, and how a released DrugWise/GMI scoring baseline behaves when its inputs are reconstructed. It covers ASD and RCSB structure discovery, ligand and target identity checks, ChEMBL and BindingDB label recovery, and PDBbind v2020 model experiments. An independently validated ASD-specific affinity predictor has not been established.

The external scoring method is described by Le and colleagues in [*Multi-level, multi-body atomic interaction graphs for machine learning-based prediction of protein-ligand binding energies*](https://doi.org/10.64898/2026.06.05.730001). The paper, its released implementation, and their model files are separate from this repository.

## What is here

| Area | Implemented work and saved evidence | Required material outside this repository |
| --- | --- | --- |
| Structure discovery | ASD allosteric-site rows are joined to RCSB experimental methods; a saved summary distinguishes X-ray, other resolved, and unresolved records. | ASD AS archive and RCSB method records |
| Label curation | PDBbind selected-ligand evidence, ChEMBL/BindingDB ligand–target checks, and online ASD mapping retain source and endpoint type. | ASD records, external label tables and API caches |
| Baseline diagnostics | Prepared-structure, Lorentz-kernel, pretrained-model, PDBbind retraining, and variant comparisons have saved aggregate summaries. | External DrugWise code and model, prepared structures, feature arrays |
| Proposed structural pipeline | AlphaFold structure prediction, pocket opening with SBALIGN, DiffDock poses, and independent ASD evaluation remain future work. | No completed outputs for these stages |

The saved ASD AS-archive snapshot has **3,102 annotation rows**, including **2,990 X-ray rows across 2,858 PDB IDs** ([discovery summary](results/reference/asd_xray_complexes_summary.json)). PDBbind ligand verification records **875 Tier-A annotation rows across 863 PDB IDs**; these include 325 `Kd`, 87 `Ki`, 462 `IC50`, and one other endpoint, so the Tier-A total is not a pure binding-constant set ([verification summary](results/reference/asd_pdbbind_tier_a_verification_summary.json)).

The recorded PDBbind v2020 reconstruction extracted usable features for 19,442 of 19,443 candidate rows and fit a `GradientBoostingRegressor` on 19,157 general-set rows. Its 285-row CASF/core test Pearson correlation is **0.8674** ([run summary](results/reference/pdbbind_plus_drugwise_retrain_summary.json)). This is a PDBbind experiment; the 21 ASD annotations used in the overlap checks represent only 20 PDB IDs already present in the paper/core test data ([variant summary](results/reference/pdbbind_kernel_variant_model_evaluation_summary.json)). Those overlap scores are diagnostics, not an independent ASD benchmark.

## Code and inputs

`scripts/` provides the named entry points for discovery, verification, inference, and PDBbind experiments. Shared discovery and identity logic is under `src/allosteric_affinity/`; model-specific scripts retain their distinct structure and evaluation paths while using a shared external DrugWise adapter. [Methods](docs/methods.md) describes the populations, evidence tiers, feature variants, and evaluation boundaries.

The Python dependencies are declared in `requirements.txt`. Metadata discovery needs standard-library tooling and, for online queries, `requests`; chemical identity checks use RDKit. The external descriptor path also needs the separate DrugWise implementation, Biopandas, and Open Babel. Saved scikit-learn models use the listed `scikit-learn==1.5.1` compatibility pin and `joblib`. NumPy and pandas are imported directly by the model workflows. The reference-paper appendix parser also requires a local `pdftotext` executable. No models, checkpoints, raw archives, or generated predictions are included.

For a local ASD AS archive and a populated RCSB method cache, the X-ray entry point can write its results inside `data/interim/`:

```powershell
python scripts/asd_discover_xray_complexes.py --archive data/raw/asd/archives/ASD_Release_202309_AS.tar.gz --methods-out data/interim/asd/rcsb_pdb_methods.tsv --xray-out data/interim/asd/asd_xray_complexes.tsv --summary-out data/interim/asd/xray_summary.json
```

If RCSB records are missing, `--online` explicitly permits the requests. Other discovery commands likewise require `--online` for remote acquisition when a needed cache is absent. The external data and software listed above must be supplied before their respective entry points can run. Discovery summaries default to ignored `outputs/discovery/` paths; the JSON files in `results/reference/` are dated records and are not default output targets.

ASD's download notice restricts third-party redistribution of its data. Raw ASD archives, derived row-level tables, external source checkouts, pretrained models, and feature matrices are therefore kept outside the public tree. The included JSON snapshots contain aggregate recorded results; their dates, population definitions, and limitations matter when comparing them.
