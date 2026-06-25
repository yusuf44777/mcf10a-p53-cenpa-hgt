from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config


ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/id"
GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def _post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_ensembl_symbols(gene_ids: list[str], chunk_size: int = 800) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(gene_ids), chunk_size):
        chunk = gene_ids[start : start + chunk_size]
        payload = {"ids": chunk}
        data = _post_json(ENSEMBL_LOOKUP_URL, payload)
        for gene_id in chunk:
            record = data.get(gene_id) or {}
            rows.append(
                {
                    "gene_id": gene_id,
                    "gene_symbol": record.get("display_name", gene_id),
                    "biotype": record.get("biotype", ""),
                    "description": record.get("description", ""),
                    "seq_region_name": record.get("seq_region_name", ""),
                }
            )
        time.sleep(0.15)
    return pd.DataFrame(rows)


def fetch_gprofiler_enrichment(gene_ids: list[str], organism: str = "hsapiens") -> pd.DataFrame:
    payload = {
        "organism": organism,
        "query": gene_ids,
        "sources": ["GO:BP", "REAC"],
        "user_threshold": 0.05,
        "no_evidences": True,
    }
    data = _post_json(GPROFILER_URL, payload)
    return pd.DataFrame(data.get("result", []))


def annotate_results(config_path: str, top_n: int = 120) -> None:
    config = load_config(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    candidate_path = results_dir / "candidate_gene_scores.csv"
    if not candidate_path.exists():
        raise FileNotFoundError("Run the main pipeline before external annotation.")
    candidates = pd.read_csv(candidate_path)
    gene_ids = candidates["gene_id"].drop_duplicates().astype(str).tolist()
    status: dict[str, Any] = {"status": "started", "n_genes": len(gene_ids)}
    try:
        annotations = fetch_ensembl_symbols(gene_ids)
        annotations.to_csv(results_dir / "gene_annotations.tsv", sep="\t", index=False)
        merged = candidates.merge(annotations, on="gene_id", how="left")
        merged.to_csv(results_dir / "candidate_gene_scores_annotated.csv", index=False)

        top_symbols = (
            merged.sort_values("candidate_score", ascending=False)
            .head(top_n)["gene_symbol"]
            .dropna()
            .astype(str)
            .tolist()
        )
        enrichment = fetch_gprofiler_enrichment(top_symbols)
        enrichment.to_csv(results_dir / "candidate_enrichment_gprofiler.csv", index=False)
        status.update(
            {
                "status": "complete",
                "annotation_source": ENSEMBL_LOOKUP_URL,
                "enrichment_source": GPROFILER_URL,
                "n_enrichment_terms": int(len(enrichment)),
            }
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        status.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        with (results_dir / "external_annotation_status.json").open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--top-n", type=int, default=120)
    args = parser.parse_args()
    annotate_results(args.config, top_n=args.top_n)


if __name__ == "__main__":
    main()
