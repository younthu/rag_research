"""BGE-M3 Evaluation on MIRACL Dataset.

This module provides evaluation of BGE-M3 embedding model on the MIRACL benchmark.
It computes standard retrieval metrics including Recall@k, MRR, and nDCG.

Usage:
    python -m miracl.bge_m3_eval --language en --split dev
    python -m miracl.bge_m3_eval --language zh --split dev
    python -m miracl.bge_m3_eval --language en zh --split dev  # Evaluate both
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_topics(topics_path: Path) -> dict[str, str]:
    """Load topics (queries) from JSONL file.
    
    Returns:
        Dict mapping query_id to query text.
    """
    topics = {}
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            topic = json.loads(line)
            topics[topic["query_id"]] = topic["query"]
    logger.info(f"Loaded {len(topics)} topics from {topics_path}")
    return topics


def load_qrels(qrels_path: Path) -> dict[str, dict[str, dict]]:
    """Load qrels (relevance judgments) from JSONL file.
    
    Returns:
        Dict mapping query_id to dict of docid -> {relevance, title, text}.
    """
    qrels = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            qrel = json.loads(line)
            query_id = qrel["query_id"]
            docid = qrel["docid"]
            qrels[query_id][docid] = {
                "relevance": qrel["relevance"],
                "title": qrel.get("title", ""),
                "text": qrel.get("text", ""),
            }
    logger.info(f"Loaded qrels for {len(qrels)} queries from {qrels_path}")
    return dict(qrels)


def get_positive_passages(qrels: dict[str, dict[str, dict]]) -> dict[str, list[dict]]:
    """Extract positive (relevant) passages for each query.
    
    Returns:
        Dict mapping query_id to list of relevant passage dicts.
    """
    positives = {}
    for query_id, docs in qrels.items():
        positives[query_id] = [
            {"docid": docid, **doc_info}
            for docid, doc_info in docs.items()
            if doc_info["relevance"] > 0
        ]
    return positives


def compute_recall_at_k(retrieved_docids: list[str], relevant_docids: set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant_docids:
        return 0.0
    retrieved_at_k = set(retrieved_docids[:k])
    return len(retrieved_at_k & relevant_docids) / len(relevant_docids)


def compute_mrr(retrieved_docids: list[str], relevant_docids: set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    for i, docid in enumerate(retrieved_docids):
        if docid in relevant_docids:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_docids: list[str], qrel_docs: dict[str, dict], k: int) -> float:
    """Compute nDCG@k."""
    def dcg(relevances: list[float]) -> float:
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    # Get relevance scores for retrieved docs
    retrieved_rels = []
    for docid in retrieved_docids[:k]:
        if docid in qrel_docs:
            retrieved_rels.append(float(qrel_docs[docid]["relevance"]))
        else:
            retrieved_rels.append(0.0)
    
    # Compute ideal DCG (all relevant docs ranked first)
    ideal_rels = sorted([float(d["relevance"]) for d in qrel_docs.values()], reverse=True)[:k]
    
    dcg_val = dcg(retrieved_rels)
    idcg_val = dcg(ideal_rels)
    
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def evaluate_bge_m3(
    language: str,
    split: str = "dev",
    output_dir: str = "output/miracl",
    top_k_values: list[int] = [1, 5, 10, 20, 50, 100],
    batch_size: int = 32,
    max_query_length: int = 512,
    max_passage_length: int = 8192,
    use_fp16: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate BGE-M3 model on MIRACL dataset.
    
    Args:
        language: Language code (e.g., "en", "zh").
        split: Dataset split (e.g., "dev", "train").
        output_dir: Output directory for MIRACL data.
        top_k_values: List of k values for Recall@k and nDCG@k.
        batch_size: Batch size for encoding.
        max_query_length: Maximum query token length.
        max_passage_length: Maximum passage token length.
        use_fp16: Whether to use FP16 for inference.
        device: Device to use (e.g., "cuda", "cpu"). If None, auto-detect.
        
    Returns:
        Dict containing evaluation metrics.
    """
    # Import here to avoid loading model when not needed
    from FlagEmbedding import BGEM3FlagModel
    import torch
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load data
    base_path = Path(output_dir) / language
    topics_path = base_path / "qrels" / f"topics_{split}.jsonl"
    qrels_path = base_path / "qrels" / f"qrels_{split}.jsonl"
    
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    
    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    positive_passages = get_positive_passages(qrels)
    
    # Prepare queries and passages
    query_ids = list(topics.keys())
    queries = [topics[qid] for qid in query_ids]
    
    # Collect all unique passages for encoding
    # passage_id is the full docid (e.g., "6007#26")
    all_passage_ids = []
    all_passages = []
    passage_id_to_idx = {}
    
    for query_id, docs in qrels.items():
        for docid, doc_info in docs.items():
            if docid not in passage_id_to_idx:
                passage_id_to_idx[docid] = len(all_passage_ids)
                all_passage_ids.append(docid)
                # Combine title and text for passage embedding
                passage_text = doc_info["title"]
                if doc_info["text"]:
                    if passage_text:
                        passage_text += "\n" + doc_info["text"]
                    else:
                        passage_text = doc_info["text"]
                all_passages.append(passage_text)
    
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Total unique passages: {len(all_passages)}")
    
    # Load BGE-M3 model
    logger.info("Loading BGE-M3 model...")
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=use_fp16,
        device=device,
    )
    
    # Encode queries
    logger.info("Encoding queries...")
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        max_length=max_query_length,
    )["dense_vecs"]
    
    # Encode passages
    logger.info("Encoding passages...")
    passage_embeddings = model.encode(
        all_passages,
        batch_size=batch_size,
        max_length=max_passage_length,
    )["dense_vecs"]
    
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    logger.info(f"Passage embeddings shape: {passage_embeddings.shape}")
    
    # Compute similarity scores (query x passage)
    logger.info("Computing similarity scores...")
    similarity_matrix = np.dot(query_embeddings, passage_embeddings.T)
    
    # Evaluate per query
    max_k = max(top_k_values)
    
    recall_scores = {k: [] for k in top_k_values}
    mrr_scores = []
    ndcg_scores = {k: [] for k in top_k_values}
    
    for i, query_id in enumerate(query_ids):
        # Get similarity scores for this query
        sim_scores = similarity_matrix[i]
        
        # Get passage indices sorted by similarity (descending)
        sorted_indices = np.argsort(sim_scores)[::-1][:max_k]
        
        # Get retrieved docids
        retrieved_docids = [all_passage_ids[idx] for idx in sorted_indices]
        
        # Get relevant docids for this query
        relevant_docids = set(
            docid for docid, info in qrels[query_id].items()
            if info["relevance"] > 0
        )
        
        # Compute metrics
        for k in top_k_values:
            recall_scores[k].append(compute_recall_at_k(retrieved_docids, relevant_docids, k))
            ndcg_scores[k].append(compute_ndcg_at_k(retrieved_docids, qrels[query_id], k))
        
        mrr_scores.append(compute_mrr(retrieved_docids, relevant_docids))
    
    # Aggregate metrics
    results = {
        "language": language,
        "split": split,
        "model": "BAAI/bge-m3",
        "num_queries": len(queries),
        "num_passages": len(all_passages),
        "metrics": {
            "MRR": float(np.mean(mrr_scores)),
        },
    }
    
    for k in top_k_values:
        results["metrics"][f"Recall@{k}"] = float(np.mean(recall_scores[k]))
        results["metrics"][f"nDCG@{k}"] = float(np.mean(ndcg_scores[k]))
    
    return results


def print_results(results: dict[str, Any]) -> None:
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print(f"BGE-M3 Evaluation Results on MIRACL ({results['language'].upper()}, {results['split']})")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Queries: {results['num_queries']}")
    print(f"Passages: {results['num_passages']}")
    print("-" * 60)
    
    metrics = results["metrics"]
    print(f"MRR: {metrics['MRR']:.4f}")
    print()
    
    # Print Recall@k
    print("Recall@k:")
    recall_keys = sorted([k for k in metrics.keys() if k.startswith("Recall@")], 
                         key=lambda x: int(x.split("@")[1]))
    for key in recall_keys:
        k = key.split("@")[1]
        print(f"  @{k}: {metrics[key]:.4f}")
    print()
    
    # Print nDCG@k
    print("nDCG@k:")
    ndcg_keys = sorted([k for k in metrics.keys() if k.startswith("nDCG@")],
                       key=lambda x: int(x.split("@")[1]))
    for key in ndcg_keys:
        k = key.split("@")[1]
        print(f"  @{k}: {metrics[key]:.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BGE-M3 on MIRACL dataset"
    )
    parser.add_argument(
        "--language", "-l",
        nargs="+",
        default=["en"],
        help="Language code(s) to evaluate (e.g., en, zh)"
    )
    parser.add_argument(
        "--split", "-s",
        default="dev",
        help="Dataset split (default: dev)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output/miracl",
        help="Output directory for MIRACL data"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detect if not specified."
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    all_results = []
    
    for lang in args.language:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating BGE-M3 on MIRACL ({lang.upper()}, {args.split})")
        logger.info(f"{'='*60}")
        
        try:
            results = evaluate_bge_m3(
                language=lang,
                split=args.split,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
            all_results.append(results)
            print_results(results)
        except Exception as e:
            logger.error(f"Error evaluating {lang}: {e}")
            raise
    
    # Save results if requested
    if args.save_results and all_results:
        output_path = Path(args.output_dir) / "bge_m3_eval_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
