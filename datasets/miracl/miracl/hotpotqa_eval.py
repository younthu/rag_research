"""HotpotQA Evaluation with Embedding Models.

This module provides evaluation of embedding models on the HotpotQA benchmark.
It computes standard retrieval metrics including Recall@k, MRR, and nDCG.

HotpotQA is a multi-hop QA dataset where each question requires reasoning
over 2 supporting paragraphs. In the distractor setting, there are 10 total
paragraphs per question (2 gold + 8 distractors).

Usage:
    python -m miracl.hotpotqa_eval --model bge-m3
    python -m miracl.hotpotqa_eval --model qwen3
    python -m miracl.hotpotqa_eval --model bge-m3 qwen3 --split validation
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


def load_eval_data(eval_file: Path) -> list[dict]:
    """Load evaluation dataset."""
    data = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples from {eval_file}")
    return data


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    for i, pid in enumerate(retrieved_ids):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute nDCG@k with binary relevance."""
    def dcg(relevances: list[float]) -> float:
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    # Get relevance scores for retrieved docs (binary: 1 if relevant, 0 otherwise)
    retrieved_rels = [1.0 if pid in relevant_ids else 0.0 for pid in retrieved_ids[:k]]
    
    # Compute ideal DCG (all relevant docs ranked first)
    num_relevant = min(len(relevant_ids), k)
    ideal_rels = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    
    dcg_val = dcg(retrieved_rels)
    idcg_val = dcg(ideal_rels)
    
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def load_model(model_name: str, device: str | None = None):
    """Load embedding model by name."""
    import torch
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    if model_name == "bge-m3":
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
        return model, "bge-m3", device
    
    elif model_name == "qwen3":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device=device,
            trust_remote_code=True,
        )
        return model, "qwen3", device
    
    else:
        # Assume it's a HuggingFace model name
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        return model, model_name, device


def encode_texts(model, texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Encode texts using the model."""
    if model_name == "bge-m3":
        embeddings = model.encode(texts, batch_size=batch_size, max_length=8192)["dense_vecs"]
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    return embeddings


def evaluate_hotpotqa(
    subset: str = "distractor",
    split: str = "validation",
    output_dir: str = "output/hotpotqa",
    model_name: str = "bge-m3",
    top_k_values: list[int] = [1, 2, 3, 5, 10],
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate embedding model on HotpotQA dataset.
    
    Args:
        subset: HotpotQA subset ('distractor' or 'fullwiki').
        split: Dataset split ('train' or 'validation').
        output_dir: Output directory for HotpotQA data.
        model_name: Model name ('bge-m3', 'qwen3', or HuggingFace model name).
        top_k_values: List of k values for Recall@k and nDCG@k.
        batch_size: Batch size for encoding.
        device: Device to use. Auto-detect if not specified.
        
    Returns:
        Dict containing evaluation metrics.
    """
    # Load model
    model, model_short_name, device = load_model(model_name, device)
    
    # Load evaluation data
    base_path = Path(output_dir) / subset / "eval"
    eval_file = base_path / f"eval_dataset_{split}.jsonl"
    
    if not eval_file.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {eval_file}. "
            f"Run hotpotqa_eval_dataset asset first."
        )
    
    eval_data = load_eval_data(eval_file)
    
    # Collect all texts for batch encoding
    queries = []
    query_ids = []
    
    # For each query, collect its candidate passages
    all_candidates: dict[str, list[dict]] = {}
    
    for item in eval_data:
        qid = item["query_id"]
        query = item["query"]
        
        queries.append(query)
        query_ids.append(qid)
        
        # Combine relevant and distractor passages as candidates
        candidates = item["relevant_passages"] + item["distractor_passages"]
        all_candidates[qid] = candidates
    
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Average candidates per query: {sum(len(v) for v in all_candidates.values()) / len(all_candidates):.1f}")
    
    # Encode queries
    logger.info("Encoding queries...")
    query_embeddings = encode_texts(model, queries, model_short_name, batch_size)
    
    # Evaluate per query
    max_k = max(top_k_values)
    
    recall_scores = {k: [] for k in top_k_values}
    mrr_scores = []
    ndcg_scores = {k: [] for k in top_k_values}
    
    # Track performance by question type and level
    type_metrics: dict[str, dict] = defaultdict(lambda: {"mrr": [], "recall@2": []})
    level_metrics: dict[str, dict] = defaultdict(lambda: {"mrr": [], "recall@2": []})
    
    logger.info("Evaluating per query...")
    
    for i, qid in enumerate(query_ids):
        item = eval_data[i]
        candidates = all_candidates[qid]
        
        if not candidates:
            continue
        
        # Get passage texts and IDs
        passage_texts = [c["title"] + "\n" + c["text"] for c in candidates]
        passage_ids = [c["passage_id"] for c in candidates]
        relevant_ids = set(c["passage_id"] for c in item["relevant_passages"])
        
        # Encode candidate passages
        passage_embeddings = encode_texts(model, passage_texts, model_short_name, batch_size)
        
        # Compute similarity
        query_emb = query_embeddings[i:i+1]
        similarities = np.dot(query_emb, passage_embeddings.T)[0]
        
        # Rank passages by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        retrieved_ids = [passage_ids[idx] for idx in sorted_indices]
        
        # Compute metrics
        for k in top_k_values:
            recall_scores[k].append(compute_recall_at_k(retrieved_ids, relevant_ids, k))
            ndcg_scores[k].append(compute_ndcg_at_k(retrieved_ids, relevant_ids, k))
        
        mrr = compute_mrr(retrieved_ids, relevant_ids)
        mrr_scores.append(mrr)
        
        # Track by type and level
        qtype = item.get("type", "unknown")
        level = item.get("level", "unknown")
        
        type_metrics[qtype]["mrr"].append(mrr)
        type_metrics[qtype]["recall@2"].append(compute_recall_at_k(retrieved_ids, relevant_ids, 2))
        
        level_metrics[level]["mrr"].append(mrr)
        level_metrics[level]["recall@2"].append(compute_recall_at_k(retrieved_ids, relevant_ids, 2))
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Evaluated {i + 1}/{len(query_ids)} queries...")
    
    # Aggregate metrics
    results = {
        "subset": subset,
        "split": split,
        "model": model_name,
        "num_queries": len(queries),
        "metrics": {
            "MRR": float(np.mean(mrr_scores)),
        },
    }
    
    for k in top_k_values:
        results["metrics"][f"Recall@{k}"] = float(np.mean(recall_scores[k]))
        results["metrics"][f"nDCG@{k}"] = float(np.mean(ndcg_scores[k]))
    
    # Add breakdown by type and level
    results["by_type"] = {
        qtype: {
            "MRR": float(np.mean(data["mrr"])),
            "Recall@2": float(np.mean(data["recall@2"])),
            "count": len(data["mrr"]),
        }
        for qtype, data in type_metrics.items()
    }
    
    results["by_level"] = {
        level: {
            "MRR": float(np.mean(data["mrr"])),
            "Recall@2": float(np.mean(data["recall@2"])),
            "count": len(data["mrr"]),
        }
        for level, data in level_metrics.items()
    }
    
    return results


def print_results(results: dict[str, Any]) -> None:
    """Pretty print evaluation results."""
    model_name = results['model']
    print("\n" + "=" * 70)
    print(f"HotpotQA Evaluation Results ({results['subset']}, {results['split']})")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Queries: {results['num_queries']}")
    print("-" * 70)
    
    metrics = results["metrics"]
    print(f"MRR: {metrics['MRR']:.4f}")
    print()
    
    print("Recall@k:")
    recall_keys = sorted([k for k in metrics.keys() if k.startswith("Recall@")], 
                         key=lambda x: int(x.split("@")[1]))
    for key in recall_keys:
        k = key.split("@")[1]
        print(f"  @{k}: {metrics[key]:.4f}")
    print()
    
    print("nDCG@k:")
    ndcg_keys = sorted([k for k in metrics.keys() if k.startswith("nDCG@")],
                       key=lambda x: int(x.split("@")[1]))
    for key in ndcg_keys:
        k = key.split("@")[1]
        print(f"  @{k}: {metrics[key]:.4f}")
    
    # Print breakdown by type
    if "by_type" in results:
        print("\n" + "-" * 70)
        print("By Question Type:")
        for qtype, data in sorted(results["by_type"].items()):
            print(f"  {qtype}: MRR={data['MRR']:.4f}, Recall@2={data['Recall@2']:.4f} (n={data['count']})")
    
    # Print breakdown by level
    if "by_level" in results:
        print("\nBy Difficulty Level:")
        for level, data in sorted(results["by_level"].items()):
            print(f"  {level}: MRR={data['MRR']:.4f}, Recall@2={data['Recall@2']:.4f} (n={data['count']})")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on HotpotQA dataset"
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        default=["bge-m3"],
        help="Model name(s) to evaluate: bge-m3, qwen3, or HuggingFace model name"
    )
    parser.add_argument(
        "--subset", "-s",
        default="distractor",
        help="HotpotQA subset: distractor or fullwiki (default: distractor)"
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split: train or validation (default: validation)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output/hotpotqa",
        help="Output directory for HotpotQA data"
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
    
    for model_name in args.model:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_name} on HotpotQA ({args.subset}, {args.split})")
        logger.info(f"{'='*60}")
        
        try:
            results = evaluate_hotpotqa(
                subset=args.subset,
                split=args.split,
                output_dir=args.output_dir,
                model_name=model_name,
                batch_size=args.batch_size,
                device=args.device,
            )
            all_results.append(results)
            print_results(results)
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise
    
    # Save results if requested
    if args.save_results and all_results:
        output_path = Path(args.output_dir) / args.subset / "eval"
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"eval_results_{args.split}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    main()
