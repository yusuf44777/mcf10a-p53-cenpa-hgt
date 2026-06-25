from __future__ import annotations

from src.config import load_config
from src.data_io import dataset_summary, load_h5ad_dense
from src.preprocess import preprocess_dataset


def test_dataset_integrity_counts() -> None:
    config = load_config("configs/default.json")
    dataset = load_h5ad_dense(config["paths"]["h5ad"])
    summary = dataset_summary(dataset)
    assert summary["n_cells"] == 9201
    assert summary["n_genes"] == 40295
    assert len(summary["samples"]) == 6
    assert summary["p53_status"] == {"p53-DN": 4845, "p53-WT": 4356}
    assert summary["CENPA_status"] == {
        "acute_oe": 2797,
        "chronic_oe": 3489,
        "control": 2915,
    }


def test_preprocessing_keeps_tp53_and_cenpa() -> None:
    config = load_config("configs/default.json")
    dataset = load_h5ad_dense(config["paths"]["h5ad"])
    processed = preprocess_dataset(dataset, config)
    gene_ids = set(processed.selected_genes["gene_id"])
    assert "ENSG00000141510" in gene_ids
    assert "ENSG00000115163" in gene_ids
    assert len(set(processed.y["condition"][processed.train_index])) == 6
    assert len(set(processed.y["condition"][processed.test_index])) == 6
