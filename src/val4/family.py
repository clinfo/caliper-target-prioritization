from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import networkx as nx
import pandas as pd

OTHER_RARE_FAMILY = "Other_rare"


def split_mechanism_tokens(raw: object) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in text.split("|"):
        tok = token.strip().lower()
        if tok == "" or tok in seen:
            continue
        out.append(tok)
        seen.add(tok)
    return out


# Backward-compatible private name used by old H1 code/tests.
_split_mechanism_tokens = split_mechanism_tokens


def build_mechanism_to_family_map(
    family_definition: dict[str, list[str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family, mechanisms in family_definition.items():
        for mech in mechanisms:
            token = str(mech).strip().lower()
            if token == "":
                continue
            if token in mapping:
                raise ValueError(
                    f"Mechanism token {token!r} is assigned to multiple families: "
                    f"{mapping[token]!r}, {family!r}."
                )
            mapping[token] = str(family)
    return mapping


def build_edge_family_inventory(
    G: nx.MultiDiGraph,
    mechanism_to_family: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for u, v, _k, attrs in G.edges(keys=True, data=True):
        edge_id = int(attrs["edge_id"])
        mechanism_raw = attrs.get("mechanism", "")
        tokens = split_mechanism_tokens(mechanism_raw)
        listed_mechanisms = [t for t in tokens if t in mechanism_to_family]
        family_candidates = sorted({mechanism_to_family[t] for t in listed_mechanisms})
        is_unassigned = len(family_candidates) == 0
        is_multi_family = len(family_candidates) >= 2
        resolved_family = family_candidates[0] if len(family_candidates) == 1 else None
        rows.append(
            {
                "edge_id": edge_id,
                "source": str(u),
                "target": str(v),
                "polarity": int(attrs.get("polarity", 0)),
                "mechanism": mechanism_raw,
                "listed_mechanisms": "|".join(listed_mechanisms),
                "family_candidates": "|".join(family_candidates),
                "resolved_family": resolved_family,
                "is_multi_family": bool(is_multi_family),
                "is_unassigned": bool(is_unassigned),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "edge_id",
            "source",
            "target",
            "polarity",
            "mechanism",
            "listed_mechanisms",
            "family_candidates",
            "resolved_family",
            "is_multi_family",
            "is_unassigned",
        ],
    )


def parse_family_candidates(raw_family_candidates: object) -> list[str]:
    if raw_family_candidates is None:
        return []
    try:
        if pd.isna(raw_family_candidates):
            return []
    except (TypeError, ValueError):
        pass
    text = str(raw_family_candidates).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in text.split("|"):
        fam = token.strip()
        if fam and fam not in seen:
            out.append(fam)
            seen.add(fam)
    return out


def build_edge_family_sets(edge_inventory_df: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    out: dict[int, tuple[str, ...]] = {}
    for row in edge_inventory_df.itertuples(index=False):
        out[int(row.edge_id)] = tuple(parse_family_candidates(row.family_candidates))
    return out


def build_edge_family_memberships(
    edge_inventory_df: pd.DataFrame,
    multi_family_edges: str,
    *,
    allow_split_equal: bool = False,
) -> dict[int, tuple[str, ...]]:
    """Return edge_id -> family memberships after applying Val4 multi-family policy.

    This helper does not duplicate graph edges. It only determines which family
    labels downstream Val4 analyses should see for each physical edge_id.
    For split_equal, the returned tuple contains all candidate families and the
    caller must apply fractional weights.
    """
    mode = str(multi_family_edges).strip().lower()
    if mode not in {"exclude", "first_family", "split_equal", "include_all"}:
        raise ValueError(f"Invalid multi_family_edges: {multi_family_edges!r}")
    if mode == "split_equal" and not bool(allow_split_equal):
        raise ValueError("multi_family_edges='split_equal' requires allow_split_equal=True for fractional callers.")

    out: dict[int, tuple[str, ...]] = {}
    for row in edge_inventory_df.itertuples(index=False):
        eid = int(row.edge_id)
        cands = parse_family_candidates(row.family_candidates)
        if len(cands) == 0:
            out[eid] = tuple()
        elif len(cands) == 1:
            out[eid] = (cands[0],)
        elif mode == "include_all":
            out[eid] = tuple(cands)
        elif mode == "split_equal":
            out[eid] = tuple(cands)
        elif mode == "first_family":
            out[eid] = (cands[0],)
        elif mode == "exclude":
            out[eid] = tuple()
        else:  # pragma: no cover
            raise AssertionError(mode)
    return out


def build_h1_family_lookups(
    edge_inventory_df: pd.DataFrame,
    multi_family_edges: str,
) -> tuple[
    dict[int, str | None],
    dict[int, bool],
    dict[int, list[str]],
    dict[int, list[str]],
]:
    mode = str(multi_family_edges).strip().lower()
    if mode not in {"exclude", "first_family", "split_equal", "include_all"}:
        raise ValueError(f"Invalid multi_family_edges: {multi_family_edges!r}")

    edge_id_to_family: dict[int, str | None] = {}
    edge_id_is_multi_family: dict[int, bool] = {}
    edge_id_split_families: dict[int, list[str]] = {}
    edge_id_include_families: dict[int, list[str]] = {}

    for row in edge_inventory_df.itertuples(index=False):
        eid = int(row.edge_id)
        cands = parse_family_candidates(row.family_candidates)
        is_multi = len(cands) >= 2
        edge_id_is_multi_family[eid] = bool(is_multi)
        if is_multi:
            if mode == "exclude":
                edge_id_to_family[eid] = None
            elif mode == "first_family":
                edge_id_to_family[eid] = cands[0] if cands else None
            elif mode == "split_equal":
                edge_id_to_family[eid] = None
                edge_id_split_families[eid] = cands
            elif mode == "include_all":
                edge_id_to_family[eid] = None
                edge_id_include_families[eid] = cands
        else:
            edge_id_to_family[eid] = cands[0] if cands else None
    return edge_id_to_family, edge_id_is_multi_family, edge_id_split_families, edge_id_include_families


def build_family_eligibility_table(
    edge_inventory_df: pd.DataFrame,
    min_physical_edges: int,
) -> pd.DataFrame:
    min_edges = int(min_physical_edges)
    if min_edges < 1:
        raise ValueError("min_physical_edges must be >= 1")
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_physical_edges": 0, "n_single_family_edges": 0, "n_multi_family_edges": 0}
    )
    assigned_edge_ids: set[int] = set()
    for row in edge_inventory_df.itertuples(index=False):
        eid = int(row.edge_id)
        fams = parse_family_candidates(row.family_candidates)
        if not fams:
            continue
        assigned_edge_ids.add(eid)
        is_multi = len(fams) >= 2
        for fam in fams:
            counts[fam]["n_physical_edges"] += 1
            if is_multi:
                counts[fam]["n_multi_family_edges"] += 1
            else:
                counts[fam]["n_single_family_edges"] += 1
    total_assigned = len(assigned_edge_ids)
    rows: list[dict[str, object]] = []
    for fam in sorted(counts):
        n = int(counts[fam]["n_physical_edges"])
        is_primary = n >= min_edges
        rows.append(
            {
                "family": fam,
                "n_physical_edges": n,
                "edge_fraction_among_assigned": (n / total_assigned) if total_assigned > 0 else 0.0,
                "n_single_family_edges": int(counts[fam]["n_single_family_edges"]),
                "n_multi_family_edges": int(counts[fam]["n_multi_family_edges"]),
                "is_primary_target_family": bool(is_primary),
                "reason": "primary_target_family" if is_primary else f"diagnostic_only:n_physical_edges<{min_edges}",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "family",
            "n_physical_edges",
            "edge_fraction_among_assigned",
            "n_single_family_edges",
            "n_multi_family_edges",
            "is_primary_target_family",
            "reason",
        ],
    )


def aggregate_rare_families_as_other(
    edge_inventory_df: pd.DataFrame,
    family_eligibility_df: pd.DataFrame,
    *,
    rare_family_name: str = OTHER_RARE_FAMILY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map families below min_physical_edges into a single analysis family.

    The input eligibility table is computed from the original mechanism families.
    This function keeps those diagnostic rows, adds an aggregate Other_rare row,
    and returns an edge inventory whose family_candidates are the analysis-time
    family labels used by H1/H2/H3.
    """
    if family_eligibility_df is None:
        return edge_inventory_df.copy(), family_eligibility_df
    if len(family_eligibility_df) == 0:
        return edge_inventory_df.copy(), family_eligibility_df.copy()

    if rare_family_name in set(family_eligibility_df["family"].astype(str)):
        raise ValueError(f"{rare_family_name!r} is reserved for aggregated rare Val4 families.")

    rare_families = set(
        family_eligibility_df.loc[
            ~family_eligibility_df["is_primary_target_family"].astype(bool),
            "family",
        ].astype(str)
    )
    if not rare_families:
        return edge_inventory_df.copy(), family_eligibility_df.copy()

    out = edge_inventory_df.copy()
    original_candidates = []
    analysis_candidates = []
    resolved_families = []
    is_multi_values = []
    is_unassigned_values = []
    for raw in out["family_candidates"].tolist():
        cands = parse_family_candidates(raw)
        original_candidates.append("|".join(cands))
        mapped = sorted({rare_family_name if fam in rare_families else fam for fam in cands})
        analysis_candidates.append("|".join(mapped))
        resolved_families.append(mapped[0] if len(mapped) == 1 else None)
        is_multi_values.append(len(mapped) >= 2)
        is_unassigned_values.append(len(mapped) == 0)

    out["original_family_candidates"] = original_candidates
    out["family_candidates"] = analysis_candidates
    out["resolved_family"] = resolved_families
    out["is_multi_family"] = is_multi_values
    out["is_unassigned"] = is_unassigned_values

    aggregate_table = build_family_eligibility_table(out, min_physical_edges=1)
    rare_row = aggregate_table.loc[aggregate_table["family"].astype(str) == rare_family_name]
    eligibility = family_eligibility_df.copy()
    rare_mask = eligibility["family"].astype(str).isin(rare_families)
    eligibility.loc[rare_mask, "reason"] = eligibility.loc[rare_mask, "reason"].astype(str).map(
        lambda reason: f"aggregated_into:{rare_family_name};{reason}"
    )
    if len(rare_row) > 0:
        row = rare_row.iloc[0].to_dict()
        row["is_primary_target_family"] = True
        row["reason"] = "primary_target_family:aggregated_rare_families"
        eligibility = pd.concat([eligibility, pd.DataFrame([row])], ignore_index=True)
    return out, eligibility


def get_primary_target_families(family_eligibility_df: pd.DataFrame) -> list[str]:
    if family_eligibility_df is None or len(family_eligibility_df) == 0:
        return []
    sub = family_eligibility_df[family_eligibility_df["is_primary_target_family"].astype(bool)]
    return [str(x) for x in sub["family"].tolist()]


def shortest_path_family_sign_counts(
    G: nx.MultiDiGraph,
    source: str,
    k_star: int,
    start_sign: int,
    edge_id_to_family: dict[int, str | None],
    edge_id_is_multi_family: dict[int, bool],
    edge_id_split_families: dict[int, list[str]],
    edge_id_include_families: dict[int, list[str]] | None = None,
) -> tuple[
    dict[str, int],
    dict[tuple[str, str], dict[int, float]],
    dict[tuple[str, str], int],
]:
    if source not in G:
        return {}, {}, {}

    dist = {
        str(node): int(d)
        for node, d in nx.single_source_shortest_path_length(G, source, cutoff=int(k_star)).items()
    }
    if len(dist) == 0:
        return {}, {}, {}

    nodes_by_dist: dict[int, list[str]] = defaultdict(list)
    for node, d in dist.items():
        nodes_by_dist[int(d)].append(node)

    state_counts: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    state_counts[str(source)][int(start_sign)] = 1.0

    family_sign_counts: dict[tuple[str, str], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    family_total_counts: dict[tuple[str, str], float] = defaultdict(float)

    for d in range(0, int(k_star)):
        for u in nodes_by_dist.get(d, []):
            sign_counts_u = state_counts.get(u)
            if not sign_counts_u:
                continue
            for _, v, _key, attrs in G.out_edges(u, keys=True, data=True):
                v = str(v)
                if v not in dist or dist[v] != d + 1:
                    continue
                pol = int(attrs.get("polarity", 0))
                if pol == 0:
                    continue
                edge_id = int(attrs["edge_id"])
                resolved_family = edge_id_to_family.get(edge_id)
                is_multi_family = bool(edge_id_is_multi_family.get(edge_id, False))
                split_fams = edge_id_split_families.get(edge_id)
                include_fams = (
                    edge_id_include_families.get(edge_id)
                    if edge_id_include_families is not None
                    else None
                )

                for sign_u, count in sign_counts_u.items():
                    sign_v = 1 if (int(sign_u) * pol) > 0 else -1
                    state_counts[v][sign_v] += float(count)
                    fc = float(count)

                    if split_fams is not None and len(split_fams) > 0:
                        w = fc / float(len(split_fams))
                        for fam in split_fams:
                            fam_key = (v, str(fam))
                            family_sign_counts[fam_key][sign_v] += w
                            family_total_counts[fam_key] += w
                    elif include_fams is not None and len(include_fams) > 0:
                        for fam in include_fams:
                            fam_key = (v, str(fam))
                            family_sign_counts[fam_key][sign_v] += fc
                            family_total_counts[fam_key] += fc
                    elif resolved_family is not None:
                        fam_key = (v, resolved_family)
                        family_sign_counts[fam_key][sign_v] += fc
                        family_total_counts[fam_key] += fc
                    elif is_multi_family:
                        pass
                    else:
                        pass

    return dist, family_sign_counts, family_total_counts
