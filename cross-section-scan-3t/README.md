# Cross-Section Scan for `pp -> tt\bar{t}`

This folder mirrors the workflow used in `cross-section-scan-2t`, but adapted to the local MadGraph process:

```text
/home/patricio/Documents/mg5amcnlo-3.x/bin/xs-signal-bsm-pp-tttbar
```

It is meant to scan the pseudoscalar mass while keeping the generated MadGraph process fixed, then save and plot the resulting cross sections.

## Contents

- `scanning/`
  Scan scripts that launch MadGraph runs over a mass grid.

- `results/`
  Saved summary tables in `csv`, `tsv`, or `json` format.

- `plotting/`
  Plotting scripts for the saved scan tables.

- `figures/`
  Saved plots.

## Assumptions

This setup assumes that:

- a local MadGraph5_aMC@NLO installation already exists
- the process directory `xs-signal-bsm-pp-tttbar` has already been generated locally
- the process lives under your MadGraph `bin/` directory

The scan scripts do not generate the process for you. They only relaunch the already prepared local process with modified parameters.

## Single Scan

From the repository root:

```bash
python3 cross-section-scan-3t/scanning/cpodd-mass-vs-cs.py \
  --mg5-bin ~/Documents/mg5amcnlo-3.x/bin \
  --process xs-signal-bsm-pp-tttbar \
  --tanphi 1 \
  --output-name bsm-tttbar-tanphi-1 \
  --mass-start 200 \
  --mass-stop 1400 \
  --mass-step 100
```

By default, the summary table is written under `cross-section-scan-3t/results/`.

## Standard `tanphi` Study

To reproduce the common `tanphi` grid used in the `2t` study (`0.01`, `0.1`, `1`, `10`, `100`):

```bash
python3 cross-section-scan-3t/scanning/run-standard-tanphi-scan.py \
  --mg5-bin ~/Documents/mg5amcnlo-3.x/bin \
  --process xs-signal-bsm-pp-tttbar \
  --mass-start 200 \
  --mass-stop 1400 \
  --mass-step 100
```

This writes files like:

- `cross-section-scan-3t/results/bsm-tttbar-tanphi-001.csv`
- `cross-section-scan-3t/results/bsm-tttbar-tanphi-01.csv`
- `cross-section-scan-3t/results/bsm-tttbar-tanphi-1.csv`
- `cross-section-scan-3t/results/bsm-tttbar-tanphi-10.csv`
- `cross-section-scan-3t/results/bsm-tttbar-tanphi-100.csv`

## Plotting

To overlay the standard `tanphi` scans in a single figure:

```bash
python3 cross-section-scan-3t/plotting/plot_tanphi_scan.py
```

By default, the figure is written to:

```text
cross-section-scan-3t/figures/bsm-tttbar-tanphi-comparison.png
```

The comparison plot uses a logarithmic y axis by default. You can switch to a linear scale with `--linear-y`.

You can also pass one or more custom CSV files through `--inputs`.
