# Event Generation

This note explains how to generate events and then combine them consistently into the same histograms, even when the production cross section and the initial number of requested events are different.

## 1. Place the UFO model

Place the UFO model in your local MadGraph installation, for example:

```text
.../mg5amcnlo-3.x/models/top-pseudoscalar-varI-BM1_UFO
```

## 2. Start MadGraph

In `.../mg5amcnlo-3.x/bin`, open a terminal and run:

```bash
./mg5amc
```

## 3. Inside the MadGraph terminal

### 3.1 Import the model

```text
import model top-pseudoscalar-varI-BM1_UFO
```

### 3.2 Define the final-state leptons

```text
l+ = e+ mu+
vl = ve vm
```

### 3.3 Generate the process

I generated the inclusive process up to the lepton level because of its subdominant branching fractions.

```text
generate p p > t t ...
```

### 3.4 Generate an output folder

```text
output tt-process
```

### 3.5 Execute the run

```text
launch tt-process
```

### 3.6 Configure the run

When MadGraph asks about the configurations:

- Turn on Pythia with `1`
- Turn on Delphes with `2`
- Turn off MadAnalysis with `3`, since we are making our own analysis

### 3.7 Modify the cards

When MadGraph asks whether you want to modify the `param_card` or `run_card`, use:

```text
set mass 36 200
set mass tanphi 1 60
set width 36 Auto
```

It is important to set the width of particle `36` (the pseudoscalar) to `Auto`, so the decays are calculated on the fly. This matters because the pseudoscalar can decay into tops, and the corresponding branching fraction is important for understanding the results.

## 4. Individual Histograms

The result of step 3 should appear in:

```text
mg5amcnlo-3.x/bin/tt-process/Events/run_01
```

This output contains `.lhe` files at the MadGraph level, `.hepmc` files at the Pythia level, and `.root` files at the Delphes level, among others.

For the histogram analysis, we are mainly interested in reading the `.root` files. For that, we use the Python package `uproot`:

```text
https://masonproffitt.github.io/uproot-tutorial/
```

If you place `event-selection-and-histograms.py` inside that folder and run it, it will generate histograms for each individual event sample.

You can choose three modes:

- Monte Carlo event mode, using unweighted events
- Monte Carlo events multiplied by cross section divided by total number of events, using weighted events
- The same as the second mode, but multiplied by a luminosity in inverse femtobarns to give the expected physical number of events

## 5. Total Histogram

If you place the outputs of the three different processes, meaning the `.lhe` and `.root` files, in folders called `run_aa`, `run_at`, and `run_tt`, and then place and run `full-histograms.py` above them, you can select and plot the combined events.

Mode `1` does not really make sense in this case. Modes `2` and `3` are the relevant ones.
