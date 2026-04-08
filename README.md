# Top Quark Signatures in 331

This repository contains the model files, scan scripts, and output tables used to study top-quark signatures in a 331 scenario with a pseudoscalar state.

The main idea is simple:

1. Define the model in FeynRules.
2. Export the model to UFO format.
3. Import the UFO model into MadGraph.
4. Generate a process once in MadGraph.
5. Run mass scans with the Python script in this repository.

This README is written for a first-year PhD student who may be using FeynRules and MadGraph locally for the first time.

## What Is In This Repository?

- `model-generation/`
  Contains the FeynRules model files, the Mathematica notebook used to load the model, and reference UFO exports for the benchmark points.

- `scan-in-madgraph/`
  Contains the Python script that runs a mass scan using an already prepared local MadGraph process.

- `results/`
  Contains summary tables produced by the scan script.

- `figures/`
  Contains plots made from the scan results.

## Important Software Note

Use **Mathematica 13** to run the notebooks in this project.

- Mathematica 13 works with this workflow.
- Mathematica 14 does **not** work reliably for this project.
- If the notebook fails in Mathematica 14, switch to Mathematica 13 instead of debugging the notebook first.

This is important enough to repeat:

**Run the FeynRules notebook with Mathematica 13, not Mathematica 14.**

## Repository Layout

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

The `_UFO/` directories in this repository are reference copies. They are useful for version control and inspection, but the real workflow is expected to happen in your **local FeynRules and MadGraph installations**.

## Before You Start

You should have the following installed on your machine:

- Mathematica 13
- FeynRules in a local directory
- MadGraph5_aMC@NLO in a local directory
- Python 3

This repository does **not** install FeynRules or MadGraph for you.

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

You must change this path so that it matches the location of **your** local FeynRules installation.

This is one of the most common sources of errors.

### 5. Make sure the notebook points to the correct model file

The notebook loads a `.fr` model file. If you want BM1, BM2, or BM3, make sure the notebook is using the correct corresponding file.

For example:

- BM1 -> `top-pseudoscalar-varI-BM1.fr`
- BM2 -> `top-pseudoscalar-varI-BM2.fr`
- BM3 -> `top-pseudoscalar-varI-BM3.fr`

If needed, edit the notebook so the `LoadModel[...]` command loads the benchmark you actually want.

## Running FeynRules With Mathematica 13

Open the notebook `load-top-pseudoscalar.nb` using **Mathematica 13**.

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

After exporting the UFO model from FeynRules, move or copy that UFO directory to a place accessible from your local MadGraph workflow.

In practice, the important point is that when you open MadGraph, the command

```text
import model top-pseudoscalar-varI-BM3_UFO
```

must work.

### 2. Start MadGraph

For example:

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
generate p p > a t
```

or any other process you want to study.

### 5. Output the process directory

Example:

```text
output varI-BM3-at
```

This creates a local MadGraph process directory with that name.

This step is important because the Python scan script does **not** generate the process for you. The script assumes the process directory already exists.

## Recommended Workflow

The cleanest workflow is:

1. Export the UFO model from FeynRules using Mathematica 13.
2. Open MadGraph locally.
3. Import the UFO model.
4. Generate the process you want.
5. Run `output ...` once to create the process directory.
6. Leave MadGraph.
7. Run the scan script from this repository.

## Running the Mass Scan

From the root of this repository:

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py
```

By default, the script assumes:

- MadGraph `bin` directory: `/home/patricio/Documents/mg5amcnlo-3.x/bin`
- Process directory name: `varI-BM3-tt`
- PDG code to modify: `36`
- Mass grid: `500, 1000, 1500`
- Output file name: `results/top-pseudoscalar-scan.csv`

If your local paths are different, you should pass the correct values through the command line options.

## What the Scan Script Does

The script:

1. Uses your local MadGraph installation.
2. Enters an already generated MadGraph process.
3. Changes the mass of the particle with the chosen PDG code.
4. Launches one run for each mass point.
5. Reads the cross section and uncertainty from MadGraph output files.
6. Saves a compact summary table in this repository.

The script does **not**:

- install MadGraph;
- install FeynRules;
- run Mathematica;
- export the UFO model;
- import the model into MadGraph for the first time;
- generate the process directory from scratch.

## Most Useful Command-Line Options

You can see all options with:

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py --help
```

The most useful ones are:

- `--mg5-bin`
  Path to your local MadGraph `bin` directory.

- `--process`
  Name of the already generated MadGraph process directory.

- `--pdg`
  PDG code of the particle whose mass you want to scan.

- `--output-name`
  Base name of the summary file written into `results/`.

- `--output-format`
  Output format: `csv`, `tsv`, or `json`.

- `--masses`
  Explicit list of masses.

- `--mass-start`, `--mass-stop`, `--mass-step`
  A regular mass grid.

- `--dry-run`
  Shows the resolved configuration without launching MadGraph.

- `--stop-on-error`
  Stops immediately if one mass point fails.

## Examples

### Example 1: Regular mass grid

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py \
  --mg5-bin ~/Documents/mg5amcnlo-3.x/bin \
  --process varI-BM3-at \
  --output-name varI-BM3-at-200-1400 \
  --mass-start 200 \
  --mass-stop 1400 \
  --mass-step 100
```

### Example 2: Explicit mass list

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py \
  --mg5-bin ~/Documents/mg5amcnlo-3.x/bin \
  --process varI-BM3-tt \
  --output-name varI-BM3-tt-custom \
  --masses 300 500 700 900 1100
```

### Example 3: Check the setup before launching runs

```bash
python3 scan-in-madgraph/cpodd-mass-vs-cs.py \
  --mg5-bin ~/Documents/mg5amcnlo-3.x/bin \
  --process varI-BM3-tt \
  --dry-run
```

## Understanding `--process` vs `--output-name`

These two options are easy to confuse.

- `--process`
  Refers to the name of the MadGraph process directory that already exists locally.

- `--output-name`
  Refers only to the name of the summary file written in this repository.

Example:

- If you ran `output varI-BM3-tt` inside MadGraph, then `--process` must be `varI-BM3-tt`.
- If you choose `--output-name my-scan`, the script will write `results/my-scan.csv`.

## Output Table

The final table contains:

- `mass`
- `run_name`
- `cross_section_pb`
- `cross_section_error_pb`
- `status`
- `return_code`
- `note`

## Common Beginner Mistakes

If something does not work, check these first:

- You opened the notebook in Mathematica 14 instead of Mathematica 13.
- The path stored in `$FeynRulesPath` is still the old local path.
- The notebook is loading the wrong benchmark `.fr` file.
- The UFO model was not exported successfully.
- MadGraph cannot find the UFO model during `import model`.
- The scan script is using the wrong `--mg5-bin` path.
- The scan script is using a `--process` name that does not match the MadGraph output directory.

## Final Practical Advice

If you are new to this workflow, do one full test by hand before launching a long scan:

1. Export one UFO model.
2. Import it into MadGraph.
3. Generate one process.
4. Run one MadGraph test.
5. Then run the Python scan script.

That makes debugging much easier than trying to debug the whole chain at once.
