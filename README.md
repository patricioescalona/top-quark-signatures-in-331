# Top Quark Signatures in 331

This repository collects the model files, cross-section scans, histogram tools, and reference outputs used to study top-quark signatures in a 331 scenario with a pseudoscalar state.

The workflow is organized by folder:

1. Build or inspect the model in `model-generation/`
2. Run cross-section scans in `cross-section-scan/`
3. Generate and combine event histograms in `histograms/`

## Repository Structure

- `model-generation/`
  FeynRules model files, the Mathematica notebook, and reference UFO exports.
  Documentation: [model-generation/README.md](/home/patricio/Documents/top-quark-signatures-in-331/model-generation/README.md)

- `cross-section-scan/`
  Scripts and outputs for MadGraph-based cross-section scans.
  Documentation: [cross-section-scan/README.md](/home/patricio/Documents/top-quark-signatures-in-331/cross-section-scan/README.md)

- `histograms/`
  Event-selection and histogram workflow.
  Documentation: [histograms/RADME.md](/home/patricio/Documents/top-quark-signatures-in-331/histograms/EVENT-GENERATION.md)

- `ckm/`
  Small helper scripts for CKM-related work.

## General Notes

- Use Mathematica 13 for the FeynRules notebook workflow.
- FeynRules and MadGraph are expected to be installed locally on your machine.
- The repository stores scripts, reference exports, plots, and tables, but the actual FeynRules and MadGraph runs happen in your local installations.

## Suggested Order

1. Start with [model-generation/README.md](/home/patricio/Documents/top-quark-signatures-in-331/model-generation/README.md) to export or inspect the UFO model.
2. Continue with [cross-section-scan/README.md](/home/patricio/Documents/top-quark-signatures-in-331/cross-section-scan/README.md) to generate MadGraph processes and run mass scans.
3. Use [histograms/EVENT-GENERATION.md](/home/patricio/Documents/top-quark-signatures-in-331/histograms/EVENT-GENERATION.md) for event production and histogram building.
