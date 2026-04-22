# Top Quark Signatures in 331

This repository contains the model files, scan scripts, and output tables used to study top-quark signatures in a 331 scenario with a pseudoscalar state.

The main idea is simple:

1. Define the model in FeynRules.
2. Export the model to UFO format.
3. Import the UFO model into MadGraph.
4. Generate a process once in MadGraph.
5. Run mass scans with the Python scripts in this repository.

This README is written for a first-year PhD student who may be using FeynRules and MadGraph locally for the first time.

## Folder Structure

- `model-generation/`
  FeynRules model files, the Mathematica notebook used to load the model, and reference UFO exports for the benchmark points.

- `cross-section-scan/scanning/`
  Python script used to run mass scans with an existing local MadGraph process.

- `cross-section-scan/results/`
  Saved scan tables in `csv`, `tsv`, or `json` format.

- `cross-section-scan/plotting/`
  Python script used to plot the scan outputs.

- `cross-section-scan/figures/`
  Saved plots.

- `ckm/`
  Small helper scripts for CKM-related work.

## Important Software Note

Use **Mathematica 13** to run the notebooks in this project.

- Mathematica 13 works with this workflow.
- Mathematica 14 does **not** work reliably for this project.
- If the notebook fails in Mathematica 14, switch to Mathematica 13 instead of debugging the notebook first.

This is important enough to repeat:

**Run the FeynRules notebook with Mathematica 13, not Mathematica 14.**

## Model Generation

Inside `model-generation/` you will find:

- `top-pseudoscalar-varI-BM1.fr`
- `top-pseudoscalar-varI-BM2.fr`
- `top-pseudoscalar-varI-BM3.fr`
- `load-top-pseudoscalar.nb`
- `top-pseudoscalar-varI-BM1_UFO/`
- `top-pseudoscalar-varI-BM2_UFO/`
- `top-pseudoscalar-varI-BM3_UFO/`

The `.fr` files are the FeynRules model definitions.

The notebook `load-top-pseudoscalar.nb` is used to load the model inside Mathematica and export the UFO model.

The `_UFO/` directories in this repository are reference copies. They are useful for version control and inspection, but the real workflow is expected to happen in your local FeynRules and MadGraph installations.

## Before You Start

You should have the following installed on your machine:

- Mathematica 13
- FeynRules in a local directory
- MadGraph5_aMC@NLO in a local directory
- Python 3

This repository does not install FeynRules or MadGraph for you.

## Local Installation: FeynRules

This section is intentionally explicit because this is the step that is usually confusing the first time.

### 1. Install FeynRules locally

You should have a local FeynRules directory, for example:

```bash
/home/your-user/Documents/feynrules-current
```

Inside that directory, FeynRules usually has a `Models/` folder.

### 2. Create a model folder inside FeynRules

Inside your local FeynRules installation, create a folder for this model, for example:

```bash
/home/your-user/Documents/feynrules-current/Models/top-pseudoscalar
```

### 3. Copy the model files into your local FeynRules installation

From this repository, copy:

- `model-generation/top-pseudoscalar-varI-BM1.fr`
- `model-generation/top-pseudoscalar-varI-BM2.fr`
- `model-generation/top-pseudoscalar-varI-BM3.fr`
- `model-generation/load-top-pseudoscalar.nb`

into your local working area in FeynRules.

For example, if you want to export benchmark point BM3, place:

- `top-pseudoscalar-varI-BM3.fr`
- `load-top-pseudoscalar.nb`

in the model folder of your local FeynRules installation.

### 4. Check the notebook path before running

In the notebook, there is a line like:

```mathematica
$FeynRulesPath = SetDirectory["/home/patricio/Documents/feynrules-current"]
```

You must change this path so that it matches the location of your local FeynRules installation.

This is one of the most common sources of errors.

### 5. Make sure the notebook points to the correct model file

The notebook loads a `.fr` model file. If you want BM1, BM2, or BM3, make sure the notebook is using the correct corresponding file.

For example:

- BM1 -> `top-pseudoscalar-varI-BM1.fr`
- BM2 -> `top-pseudoscalar-varI-BM2.fr`
- BM3 -> `top-pseudoscalar-varI-BM3.fr`

If needed, edit the notebook so the `LoadModel[...]` command loads the benchmark you actually want.

## Running FeynRules With Mathematica 13

Open the notebook `load-top-pseudoscalar.nb` using Mathematica 13.

Then run the notebook cells step by step:

1. Set the FeynRules path.
2. Load `FeynRules``.
3. Go to the model directory.
4. Load the model with `LoadModel[...]`.
5. Export the UFO model.

Do not skip the version check:

- If you are in Mathematica 13, continue.
- If you are in Mathematica 14, stop and reopen the notebook in Mathematica 13.

## What Comes Out of FeynRules?

After a successful export, FeynRules will create a UFO model directory.

That UFO directory is what MadGraph needs.

Typical example:

- FeynRules input: `top-pseudoscalar-varI-BM3.fr`
- FeynRules output: `top-pseudoscalar-varI-BM3_UFO/`

## Local Installation: MadGraph

You should also have a local MadGraph installation, for example:

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

In practice, the important point is that when you open MadGraph, the command

```text
import model top-pseudoscalar-varI-BM3_UFO
```

must work.

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

This creates a local MadGraph process directory with that name.

This step is important because the Python scan script does not generate the process for you. The script assumes the process directory already exists.

## Recommended Workflow

The cleanest workflow is:

1. Export the UFO model from FeynRules using Mathematica 13.
2. Place the exported UFO directory inside the `models/` folder of your local MadGraph installation. For example: `/home/your-user/Documents/mg5amcnlo-3.x/models/top-pseudoscalar-varI-BM1_UFO`
3. Open a terminal in `/home/your-user/Documents/mg5amcnlo-3.x/bin` and run `./mg5_aMC`
4. Inside the MadGraph terminal, import the UFO model. For example: `import model top-pseudoscalar-varI-BM3_UFO`
5. Generate the process you want to scan. For example: `generate p p > t t`
6. Create the output directory for that process. For example: `output varI-BM3-tt`
7. Leave MadGraph.
8. Run the scan script from this repository.

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

```bash
python3 cross-section-scan/plotting/plot_scan_channels.py
```

By default, the plot is written under `cross-section-scan/figures/`.
