# Cross-Section Scan

This folder contains the scripts and outputs used to run mass scans with an already prepared local MadGraph process and to plot the resulting cross sections.

## Contents

- `scanning/`
  Scan scripts that launch MadGraph runs over a mass grid.

- `results/`
  Saved summary tables in `csv`, `tsv`, or `json` format.

- `plotting/`
  Plotting scripts for the saved scan tables.

- `figures/`
  Saved plots.

## Before Running a Scan

You should have:

- A local MadGraph5_aMC@NLO installation
- A UFO model already exported from FeynRules
- A generated MadGraph process directory that already exists locally

The scan script does not generate the process for you. It assumes the process directory already exists in your local MadGraph installation.

## Local Installation: MadGraph

You should have a local MadGraph installation, for example:

```bash
/home/your-user/Documents/mg5amcnlo-3.x
```

The executable is typically here:

```bash
/home/your-user/Documents/mg5amcnlo-3.x/bin/mg5_aMC
```

### 1. Put the UFO model where MadGraph can import it

After exporting the UFO model from FeynRules, move or copy that UFO directory into the `models/` folder of your local MadGraph installation.

For example:

```bash
/home/your-user/Documents/mg5amcnlo-3.x/models/top-pseudoscalar-varI-BM1_UFO
```

### 2. Start MadGraph

Open a terminal in the `bin/` directory of your local MadGraph installation and run:

```bash
cd ~/Documents/mg5amcnlo-3.x/bin
./mg5_aMC
```

### 3. Import the UFO model

Inside the MadGraph prompt:

```text
import model top-pseudoscalar-varI-BM3_UFO
```

Replace `BM3` by `BM1` or `BM2` if that is the benchmark you exported.

### 4. Generate the process

Example:

```text
generate p p > t t
```

or any other process you want to study.

### 5. Output the process directory

Example:

```text
output varI-BM3-tt
```

This creates the local MadGraph process directory used later by the scan script.

## Recommended Workflow

1. Export the UFO model from FeynRules.
2. Place the exported UFO directory inside the `models/` folder of your local MadGraph installation.
3. Open a terminal in `/home/your-user/Documents/mg5amcnlo-3.x/bin` and run `./mg5_aMC`.
4. Inside the MadGraph terminal, import the UFO model.
5. Generate the process you want to scan.
6. Create the output directory for that process.
7. Leave MadGraph.
8. Run the scan script from this folder.

## Running a Scan

From the repository root:

```bash
python3 cross-section-scan/scanning/cpodd-mass-vs-cs.py \
  --mg5-bin ~/Documents/mg5amcnlo-3.x/bin \
  --process varI-BM3-tt \
  --output-name varI-BM3-tt-200-1400 \
  --mass-start 200 \
  --mass-stop 1400 \
  --mass-step 100
```

By default, the output table is written under `cross-section-scan/results/`.

## Making a Plot

From the repository root:

```bash
python3 cross-section-scan/plotting/plot_scan_channels.py
```

By default, the plot is written under `cross-section-scan/figures/`.
