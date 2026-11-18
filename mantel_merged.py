"""Several Mantel tests.

1. Colony level: FST and RST between colonies vs ln(km), separately for each
   subspecies; 9999 random permutations of colony labels, two-sided p.
2. Partial Mantel: ln(distance) | "different region" and vice versa. The
   predictors are strongly correlated, so these tests are exploratory.
3. Regional level (Borena, Babile, Jeldessa, Megadu), two variants:
   a) individual level: D = 1 - r between individuals from DIFFERENT colonies;
   b) aggregated: mean D for each pair of colonies.
   In both cases all n! permutations of colony labels are enumerated (exact
   test); within-colony pairs are excluded.

All distances are great-circle (haversine) distances from ONE coordinate table.
Source files are only read.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EARTH_RADIUS_KM = 6371.0088
TOL = 1e-12  # tolerance for floating-point error when comparing |r|

DEFAULT_RELATEDNESS_DIR = Path(r"C:\!CURRENT\Heterocephalus\New var\Relatedness\data")
DEFAULT_FST_DIR = Path(r"C:\!CURRENT\Heterocephalus\New var\Fst_Rst")

SUBSPECIES = {
    "H_g_ansorgei": ["ar1", "ar3", "br1", "br3", "br4", "br6", "br7",
                     "mg1", "mg2", "mg3", "mg4", "mg5"],
    "H_g_glaber": ["dd1", "dd2", "es2", "es3", "es4", "jd1", "jd2", "jd3"],
}

# region -> relatedness file
REGIONS = {
    "Borena National Park": "rel_br.csv",
    "Babile Elephant Sanctuary": "rel_es.csv",
    "Jeldessa": "rel_jd.csv",
    "Megadu block": "rel_mg.csv",
}


# ----------------------------------------------------------------------------
# Data input
# ----------------------------------------------------------------------------

def to_float(s):
    # the files use both "," and "." as the decimal separator
    return float(s.strip().replace(",", "."))


def read_table(path, delimiter):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter=delimiter))


def symmetrize(m, diag, labels, path):
    """A matrix may be filled in one triangle only: fill in the other one."""
    for i in range(len(labels)):
        m[i, i] = diag
        for j in range(i):
            lower, upper = m[i, j], m[j, i]
            if np.isfinite(lower) and np.isfinite(upper):
                if abs(lower - upper) > 1e-6:
                    warnings.warn(f"{Path(path).name}: {labels[i]}/{labels[j]} is asymmetric "
                                  f"({lower} vs {upper}), lower triangle used")
                value = lower
            elif np.isfinite(lower):
                value = lower
            elif np.isfinite(upper):
                value = upper
            else:
                raise ValueError(f"Missing value in {path}: {labels[i]} vs {labels[j]}")
            m[i, j] = m[j, i] = value
    return m


def read_pairwise_matrix(path):
    """FST/RST matrix (tab-delimited, colony labels in the first row)."""
    rows = read_table(path, "\t")
    labels = [c.strip() for c in rows[0][1:] if c.strip() and not c.startswith("----")]
    idx = {name: i for i, name in enumerate(labels)}
    m = np.full((len(labels), len(labels)), np.nan)
    for row in rows[1:]:
        if not row or row[0].strip() not in idx:
            continue
        i = idx[row[0].strip()]
        for j, v in enumerate(row[1:len(labels) + 1]):
            if v.strip():
                m[i, j] = to_float(v)
    return labels, symmetrize(m, 0.0, labels, path)


def read_relatedness(path):
    """Column 1 = colony, column 2 = individual, then the relatedness matrix."""
    rows = read_table(path, ";")
    ids = [c.strip() for c in rows[0][2:] if c.strip()]
    idx = {name: i for i, name in enumerate(ids)}
    colony_of = {}
    m = np.full((len(ids), len(ids)), np.nan)
    for row in rows[1:]:
        if len(row) < 2 or row[1].strip() not in idx:
            continue
        ind = row[1].strip()
        colony_of[ind] = row[0].strip()
        i = idx[ind]
        for j, v in enumerate(row[2:len(ids) + 2]):
            if v.strip():
                m[i, j] = to_float(v)
    if set(ids) != set(colony_of):
        raise ValueError(f"Some individuals have no colony label in {path}")
    return ids, colony_of, symmetrize(m, 1.0, ids, path)


def read_coordinates(path, fmt):
    """colony -> (lat, lon) in degrees.

    simple   : one header row; columns colony, lat, lon (the former coordinates.tsv)
    table_s1 : two header rows; colony in column 2, lat/lon in columns 4-5 (Table_S1.tsv)
    """
    rows = read_table(path, "\t")
    if fmt == "simple":
        body, c_col, c_lat, c_lon = rows[1:], 0, 1, 2
    else:
        body, c_col, c_lat, c_lon = rows[2:], 1, 3, 4
    coords = {}
    for row in body:
        if len(row) <= max(c_col, c_lat, c_lon) or not row[c_col].strip():
            continue
        name = row[c_col].strip()
        point = (to_float(row[c_lat]), to_float(row[c_lon]))
        if name in coords and coords[name] != point:
            raise ValueError(f"Conflicting coordinates for {name}")
        coords[name] = point
    return coords


# ----------------------------------------------------------------------------
# Geometry and statistics
# ----------------------------------------------------------------------------

def haversine_km(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(h))


def geo_matrix(colonies, coords):
    missing = [c for c in colonies if c not in coords]
    if missing:
        raise ValueError(f"No coordinates for colonies: {missing}")
    n = len(colonies)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            m[i, j] = m[j, i] = haversine_km(coords[colonies[i]], coords[colonies[j]])
    return m


def ln_distance(geo, colonies):
    off = ~np.eye(len(geo), dtype=bool)
    if (geo[off] <= 0).any():
        raise ValueError(f"Different colonies in {colonies} share coordinates: ln(0) is undefined")
    out = np.zeros_like(geo)
    out[off] = np.log(geo[off])
    return out


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def partial_corr(y, x, z):
    yz, xz, yx = corr(y, z), corr(x, z), corr(y, x)
    denom = (1 - yz ** 2) * (1 - xz ** 2)
    if denom <= TOL:
        raise ValueError("Predictors are collinear: partial correlation is undefined")
    return (yx - yz * xz) / math.sqrt(denom)


def dissimilarity(rel):
    """D = 1 - r, with negative values clipped to 0."""
    d = np.clip(1 - rel, 0, None)
    np.fill_diagonal(d, 0)
    return d


def permutation_test(stat, n, *, exact, n_perm=None, seed=None):
    """Two-sided permutation test.

    stat(order) is the statistic under the permutation `order` of object labels.
    exact=True  : all n! permutations are enumerated, p = share with |stat| >= |obs|.
    exact=False : n_perm random permutations, p = (hits + 1) / (n_perm + 1).
    Returns (observed, p, number of permutations, minimum attainable p).
    """
    obs = stat(np.arange(n))
    if exact:
        values = np.array([stat(np.array(order)) for order in itertools.permutations(range(n))])
        a = np.abs(values)
        return obs, float(np.mean(a >= abs(obs) - TOL)), len(values), float(np.mean(a >= a.max() - TOL))
    rng = np.random.default_rng(seed)
    hits = sum(abs(stat(rng.permutation(n))) >= abs(obs) - TOL for _ in range(n_perm))
    return obs, (hits + 1) / (n_perm + 1), n_perm, 1 / (n_perm + 1)


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------

def colony_level(coords, args):
    """FST/RST vs ln(km) and partial Mantel tests within subspecies."""
    metrics = (
        ("FST", read_pairwise_matrix(args.fst_dir / "Fst_multilocus_colonies.txt")),
        ("RST", read_pairwise_matrix(args.fst_dir / "Rst_multilocus_colonies.txt")),
    )
    mantel_rows, partial_rows = [], []

    for gi, (group, colonies) in enumerate(SUBSPECIES.items()):
        n = len(colonies)
        tri = np.triu_indices(n, 1)
        log_geo = ln_distance(geo_matrix(colonies, coords), colonies)
        # 1 if the two colonies are from different regions (region = first two letters of the ID)
        reg = np.array([c[:2] for c in colonies])
        different = (reg[:, None] != reg[None, :]).astype(float)

        for mi, (metric, (labels, whole)) in enumerate(metrics):
            missing = [c for c in colonies if c not in labels]
            if missing:
                raise ValueError(f"Colonies missing from the {metric} matrix: {missing}")
            keep = [labels.index(c) for c in colonies]
            gen = whole[np.ix_(keep, keep)]

            def permuted(order, gen=gen, tri=tri):
                return gen[np.ix_(order, order)][tri]

            x = log_geo[tri]
            r, p, total, p_min = permutation_test(
                lambda o: corr(x, permuted(o)), n,
                exact=False, n_perm=args.n_perm, seed=args.seed + gi * 10 + mi)
            mantel_rows.append(dict(
                group=group, metric=metric, n_colonies=n, n_pairs=len(x),
                mantel_r=r, p_two_sided=p, permutations=total,
                geographic_transform="ln(km)"))

            for focal_name, control_name, focal, control, offset in (
                ("ln_geographic_distance", "different_region", log_geo, different, 100),
                ("different_region", "ln_geographic_distance", different, log_geo, 200),
            ):
                f, c = focal[tri], control[tri]
                pr, pp, total, p_min = permutation_test(
                    lambda o: partial_corr(permuted(o), f, c), n,
                    exact=False, n_perm=args.n_perm, seed=args.seed + offset + gi * 10 + mi)
                partial_rows.append(dict(
                    group=group, metric=metric, n_colonies=n,
                    focal_predictor=focal_name, controlled_for=control_name,
                    partial_r=pr, p_two_sided=pp, permutations=total))

    return mantel_rows, partial_rows


def individual_block_test(ids, colony_of, d, colonies, geo):
    """Individuals from different colonies: D vs distance between their colonies.

    Individual pairs are not independent, so permutations are done at the colony
    level (colony labels are permuted as whole blocks), with exact enumeration.
    """
    pos = {c: i for i, c in enumerate(colonies)}
    ci = np.array([pos[colony_of[x]] for x in ids])
    iu, ju = np.tril_indices(len(ids), -1)
    between = ci[iu] != ci[ju]
    a, b = ci[iu][between], ci[ju][between]
    g = d[iu, ju][between]
    result = permutation_test(lambda order: corr(geo[order[a], order[b]], g),
                              len(colonies), exact=True)
    return result, len(g)


def regional_level(coords, args):
    ind_rows, agg_rows, pair_rows = [], [], []

    for region, fname in REGIONS.items():
        ids, colony_of, rel = read_relatedness(args.relatedness_dir / fname)
        colonies = sorted(set(colony_of.values()))
        n = len(colonies)
        if n < 3:
            warnings.warn(f"{region}: fewer than 3 colonies, test skipped")
            continue
        geo = geo_matrix(colonies, coords)
        d = dissimilarity(rel)

        # a) individual level
        (r, p, total, p_min), n_pairs = individual_block_test(ids, colony_of, d, colonies, geo)
        ind_rows.append(dict(
            region=region, n_individuals=len(ids), n_colonies=n,
            n_between_colony_individual_pairs=n_pairs,
            mantel_r=r, p_two_sided_exact=p, permutations=total, min_attainable_p=p_min,
            genetic_measure="individual D=1-r clipped at 0",
            geographic_measure="distance between colonies (km)",
            permutation="exact colony geography blocks"))

        # b) aggregation over colony pairs
        members = {c: [i for i, x in enumerate(ids) if colony_of[x] == c] for c in colonies}
        mean_d = np.zeros((n, n))
        counts = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i):
                block = d[np.ix_(members[colonies[i]], members[colonies[j]])]
                mean_d[i, j] = mean_d[j, i] = block.mean()
                counts[i, j] = counts[j, i] = block.size
        tri = np.triu_indices(n, 1)
        x = geo[tri]
        r, p, total, p_min = permutation_test(
            lambda order: corr(x, mean_d[np.ix_(order, order)][tri]), n, exact=True)
        agg_rows.append(dict(
            region=region, n_individuals=len(ids), n_colonies=n, n_colony_pairs=len(x),
            mantel_r=r, p_two_sided_exact=p, permutations=total, min_attainable_p=p_min,
            genetic_measure="mean between-colony D=1-r",
            geographic_measure="distance between colonies (km)"))
        for i in range(n):
            for j in range(i):
                pair_rows.append(dict(
                    region=region, colony_1=colonies[j], colony_2=colonies[i],
                    mean_D_1_minus_r=mean_d[i, j], geographic_distance_km=geo[i, j],
                    n_individual_pairs=counts[i, j]))

    return ind_rows, agg_rows, pair_rows


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

README = """\
Colony-level tests (colony_level_FST_RST.csv): current 20 colonies, FST or RST,
ln-transformed great-circle distance (km) from one coordinate table, Pearson matrix
correlation, 9999 random permutations of colony labels, two-sided p = (hits+1)/(N+1).
Partial Mantel tests (colony_level_partial_mantel.csv): ln distance vs a same/different-region
indicator within each subspecies; the predictors are strongly confounded, so the tests are
exploratory.
Regional tests use raw km and exact enumeration of all n! colony-label permutations,
two-sided p = share of permutations with |r| >= |r_obs| (identity included).
 - regional_individual_level.csv: D=1-r (clipped at 0) between individuals from different
   colonies; within-colony pairs excluded; colony geography blocks are permuted.
 - regional_colony_aggregated.csv: mean between-colony D per colony pair.
 min_attainable_p is the smallest p the permutation scheme can give; with 3 colonies there
 are only 3! = 6 permutations, so p >= 1/6 and such tests cannot reach significance.
Arero Forest and Dire Dawa are not tested separately: only two sampled colonies each.
"""


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", type=Path, default=HERE / "coordinates.tsv",
                   help="table of colony coordinates")
    p.add_argument("--coords-format", choices=("simple", "table_s1"), default="simple",
                   help="simple: colony/lat/lon; table_s1: Table_S1.tsv layout")
    p.add_argument("--relatedness-dir", type=Path, default=DEFAULT_RELATEDNESS_DIR)
    p.add_argument("--fst-dir", type=Path, default=DEFAULT_FST_DIR)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--n-perm", type=int, default=9999)
    p.add_argument("--seed", type=int, default=20260905)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    coords = read_coordinates(args.coords, args.coords_format)

    mantel_rows, partial_rows = colony_level(coords, args)
    ind_rows, agg_rows, pair_rows = regional_level(coords, args)

    write_csv(args.out / "colony_level_FST_RST.csv", mantel_rows)
    write_csv(args.out / "colony_level_partial_mantel.csv", partial_rows)
    write_csv(args.out / "regional_individual_level.csv", ind_rows)
    write_csv(args.out / "regional_colony_aggregated.csv", agg_rows)
    write_csv(args.out / "regional_colony_pair_values.csv", pair_rows)
    (args.out / "README.txt").write_text(README, encoding="utf-8")

    for rows in (mantel_rows, partial_rows, ind_rows, agg_rows):
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
