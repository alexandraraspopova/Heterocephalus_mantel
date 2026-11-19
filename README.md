# Mantel tests for *Heterocephalus glaber* population structure

A single script, `mantel_merged.py`, that runs all Mantel analyses used in the reviewer response:

| Level | Test | Permutations |
|---|---|---|
| Colony, per subspecies | FST / RST vs ln(km) | 9999 random, two-sided |
| Colony, per subspecies | Partial Mantel: ln(km) \| different region, and vice versa (exploratory, predictors are confounded) | 9999 random, two-sided |
| Region (Borena, Babile, Jeldessa, Megadu) | Individual D = 1 - r between individuals of different colonies vs colony distance | exact, all n! colony-label permutations |
| Region | Mean between-colony D per colony pair vs colony distance | exact, all n! |

All distances are great-circle (haversine) distances computed from one coordinate table. Source files are only read.

## Requirements

Python 3.8+ and NumPy.

## Usage

```bash
python mantel_merged.py \
    --coords coordinates.tsv \
    --relatedness-dir path/to/Relatedness/data \
    --fst-dir path/to/Fst_Rst \
    --out outputs
```

| Option | Default | Meaning |
|---|---|---|
| `--coords` | `coordinates.tsv` next to the script | colony coordinates |
| `--coords-format` | `simple` | `simple`: columns colony, lat, lon (one header row); `table_s1`: layout of `Table_S1.tsv` |
| `--relatedness-dir` | hard-coded path in the script | folder with `rel_br.csv`, `rel_es.csv`, `rel_jd.csv`, `rel_mg.csv` |
| `--fst-dir` | hard-coded path in the script | folder with `Fst_multilocus_colonies.txt`, `Rst_multilocus_colonies.txt` |
| `--out` | `outputs/` next to the script | output folder |
| `--n-perm` | 9999 | random permutations for colony-level tests |
| `--seed` | 20260905 | random seed |

## Input formats

- Coordinates: tab-separated, decimal point or comma.
- FST / RST: tab-separated matrices with colony labels in the first row.
- Relatedness: `;`-separated; column 1 is the colony, column 2 the individual, then the (lower-triangular or full) relatedness matrix.

## Output (`--out`)

- `colony_level_FST_RST.csv`
- `colony_level_partial_mantel.csv`
- `regional_individual_level.csv`
- `regional_colony_aggregated.csv`
- `regional_colony_pair_values.csv`
- `README.txt` (short description of the methods)

## Notes

- Regional tests include a `min_attainable_p` column. With 3 colonies there are only 3! = 6 permutations, so p cannot be lower than 1/6 and such tests cannot reach significance.
- Colony-level tests use ln(km); regional tests use raw km.
- A colony's region is taken from the first two letters of its ID (e.g. `br1` → `br`).
