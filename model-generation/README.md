# Model Generation

This folder contains the FeynRules model files, the Mathematica notebook used to load the model, and reference UFO exports for the benchmark points.

## Contents

Inside this folder you will find:

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

## Important Software Note

Use Mathematica 13 for the notebook workflow.

- Mathematica 13 works with this setup.
- Mathematica 14 does not work reliably for this project.
- If the notebook fails in Mathematica 14, switch to Mathematica 13 first.

## Before You Start

You should have the following installed on your machine:

- Mathematica 13
- FeynRules in a local directory
- Python 3

This repository does not install FeynRules for you.

## Local Installation: FeynRules

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

### 5. Make sure the notebook points to the correct model file

The notebook loads a `.fr` model file. If you want BM1, BM2, or BM3, make sure the notebook is using the correct corresponding file.

For example:

- BM1 -> `top-pseudoscalar-varI-BM1.fr`
- BM2 -> `top-pseudoscalar-varI-BM2.fr`
- BM3 -> `top-pseudoscalar-varI-BM3.fr`

If needed, edit the notebook so the `LoadModel[...]` command loads the benchmark you actually want.

## Running FeynRules

Open the notebook `load-top-pseudoscalar.nb` using Mathematica 13.

Then run the notebook cells step by step:

1. Set the FeynRules path.
2. Load `FeynRules``.
3. Go to the model directory.
4. Load the model with `LoadModel[...]`.
5. Export the UFO model.

## Output

After a successful export, FeynRules will create a UFO model directory.

Typical example:

- FeynRules input: `top-pseudoscalar-varI-BM3.fr`
- FeynRules output: `top-pseudoscalar-varI-BM3_UFO/`

That UFO directory is what MadGraph needs in the next step of the workflow.
