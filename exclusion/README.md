# Exclusion Workflow

This folder contains the terminal workflow used to:

1. generate events with MadGraph,
2. compress the Delphes outputs into parquet files,
3. apply the cutflow and save the results to CSV files.

## Before Running

You should have:

- a local MadGraph5_aMC@NLO installation,
- the `proc-*` process folders already created inside your MadGraph `bin/` directory,
- the Python packages required by the scripts installed in your environment.

By default, the scripts expect the MadGraph `bin/` folder at:

```bash
/home/patricio/Documents/mg5amcnlo-3.x/bin
```

## 1. Generate Events and Compress Them

From the repository root, run:

```bash
python3 exclusion/generator-and-compressor.py 10000 \
  --mass 200 \
  --tanphi 60
```

This writes a MadGraph command file, launches all discovered `proc-*` folders, and saves the compressed parquet outputs under:

```bash
exclusion/generated-m-200-tanphi-60/
```

If you only want to write the `.mg5` card without launching MadGraph:

```bash
python3 exclusion/generator-and-compressor.py 10000 \
  --mass 200 \
  --tanphi 60 \
  --write-only
```

If your MadGraph installation is in a different place:

```bash
python3 exclusion/generator-and-compressor.py 10000 \
  --mass 200 \
  --tanphi 60 \
  --mg5-bin /path/to/mg5amcnlo/bin
```

## 2. Apply the Cuts

After the parquet files are created, run:

```bash
python3 exclusion/cuts.py --mass 200 --tanphi 60
```

This writes:

- `cuts-m-200-tanphi-60.csv`
- `efficiencies-m-200-tanphi-60.csv`

inside the same generated folder.

## 3. Run Over a Range

`generator-and-compressor.py` can also run a scan directly.

Masses are scanned linearly with `--mass-range`, and `tanphi` is scanned logarithmically with `--tanphi-range`.

Example:

```bash
python3 exclusion/generator-and-compressor.py 10000 \
  --mass-range 200 500 \
  --mass-points 4 \
  --tanphi-range 10 1000 \
  --tanphi-points 3
```

This runs the grid:

- masses: `200, 300, 400, 500`
- tanphi: `10, 100, 1000`

If you want a scan in only one variable, keep the other one fixed:

```bash
python3 exclusion/generator-and-compressor.py 10000 \
  --mass-range 200 500 \
  --mass-points 4 \
  --tanphi 60
```

```bash
python3 exclusion/generator-and-compressor.py 10000 \
  --mass 200 \
  --tanphi-range 1 100 \
  --tanphi-points 3
```

After the generated folders exist, `cuts.py` can process all of them automatically:

```bash
python3 exclusion/cuts.py
```

If you want to process only one generated point, use:

```bash
python3 exclusion/cuts.py --mass 200 --tanphi 60
```

## Output Layout

For one point in parameter space, the results are stored in:

```bash
exclusion/generated-m-<mass>-tanphi-<tanphi>/
```

Typical files are:

- `parquets/proc-*.parquet`
- `xsec-m-<mass>-tanphi-<tanphi>.csv`
- `cuts-m-<mass>-tanphi-<tanphi>.csv`
- `efficiencies-m-<mass>-tanphi-<tanphi>.csv`
- `decays-m-<mass>-tanphi-<tanphi>.txt`
