"""Qwen3 Embedding Evaluation on MIRACL Dataset.

This module provides evaluation of Qwen3 Embedding model on the MIRACL benchmark.
It computes standard retrieval metrics including Recall@k, MRR, and nDCG.

Usage:
    python -m miracl.qwen3_eval --language en --split dev
    python -m miracl.qwen3_eval --language zh --split dev
    python -m miracl.qwen3_eval --language en zh --split dev  # Evaluate both
    python -m miracl.qwen3_eval --model Qwen/Qwen3-Embedding-4B  # Use larger model
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
    """Load topics (queries) from JSONL file."""
    topics = {}
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            topic = json.loads(line)
            topics[topic["query_id"]] = topic["query"]
    logger.info(f"Loaded {len(topics)} topics from {topics_path}")
    return topics


def load_qrels(qrels_path: Path) -> dict[str, dict[str, dict]]:
    """Load qrels (relevance judgments) from JSONL file."""
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
    
    retrieved_rels = []
    for docid in retrieved_docids[:k]:
        if docid in qrel_docs:
            retrieved_rels.append(float(qrel_docs[docid]["relevance"]))
        else:
            retrieved_rels.append(0.0)
    
    ideal_rels = sorted([float(d["relevance"]) for d in qrel_docs.values()], reverse=True)[:k]
    
    dcg_val = dcg(retrieved_rels)
    idcg_val = dcg(ideal_rels)
    
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def evaluate_qwen3(
    language: str,
    split: str = "dev",
    output_dir: str = "output/miracl",
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    top_k_values: list[int] = [1, 5, 10, 20, 50, 100],
    batch_size: int = 32,
    max_length: int = 8192,
    device: str | None = None,
    query_instruction: str = "",
    passage_instruction: str = "",
) -> dict[str, Any]:
    """Evaluate Qwen3 Embedding model on MIRACL dataset.
    
    Args:
        language: Language code (e.g., "en", "zh").
        split: Dataset split (e.g., "dev", "train").
        output_dir: Output directory for MIRACL data.
        model_name: Qwen3 embedding model name.
        top_k_values: List of k values for Recall@k and nDCG@k.
        batch_size: Batch size for encoding.
        max_length: Maximum token length.
        device: Device to use (e.g., "cuda", "cpu"). If None, auto-detect.
        query_instruction: Instruction prefix for queries.
        passage_instruction: Instruction prefix for passages.
        
    Returns:
        Dict containing evaluation metrics.
    """
    from sentence_transformers import SentenceTransformer
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
    
    # Prepare queries and passages
    query_ids = list(topics.keys())
    queries = [topics[qid] for qid in query_ids]
    
    # Add instruction to queries if provided
    if query_instruction:
        queries = [f"{query_instruction}{q}" for q in queries]
    
    # Collect all unique passages for encoding
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
                # Add instruction to passages if provided
                if passage_instruction:
                    passage_text = f"{passage_instruction}{passage_text}"
                all_passages.append(passage_text)
    
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Total unique passages: {len(all_passages)}")
    
    # Load Qwen3 Embedding model
    logger.info(f"Loading model: {model_name}...")
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
    )
    
    # Set max sequence length
    if hasattr(model, 'max_seq_length'):
        model.max_seq_length = max_length
    
    import time
    
    # Encode queries
    logger.info("Encoding queries...")
    query_start_time = time.time()
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_encode_time = time.time() - query_start_time
    
    # Encode passages
    logger.info("Encoding passages...")
    passage_start_time = time.time()
    passage_embeddings = model.encode(
        all_passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    passage_encode_time = time.time() - passage_start_time
    
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    logger.info(f"Passage embeddings shape: {passage_embeddings.shape}")
    
    # Calculate encoding speed
    query_speed = len(queries) / query_encode_time
    passage_speed = len(all_passages) / passage_encode_time
    total_encode_time = query_encode_time + passage_encode_time
    
    logger.info(f"Query encoding: {query_encode_time:.2f}s ({query_speed:.2f} queries/sec)")
    logger.info(f"Passage encoding: {passage_encode_time:.2f}s ({passage_speed:.2f} passages/sec)")
    logger.info(f"Total encoding time: {total_encode_time:.2f}s")
    
    # Compute similarity scores (query x passage)
    # Since embeddings are normalized, dot product = cosine similarity
    logger.info("Computing similarity scores...")
    similarity_matrix = np.dot(query_embeddings, passage_embeddings.T)
    
    # Evaluate per query
    max_k = max(top_k_values)
    
    recall_scores = {k: [] for k in top_k_values}
    mrr_scores = []
    ndcg_scores = {k: [] for k in top_k_values}
    
    for i, query_id in enumerate(query_ids):
        sim_scores = similarity_matrix[i]
        sorted_indices = np.argsort(sim_scores)[::-1][:max_k]
        retrieved_docids = [all_passage_ids[idx] for idx in sorted_indices]
        
        relevant_docids = set(
            docid for docid, info in qrels[query_id].items()
            if info["relevance"] > 0
        )
        
        for k in top_k_values:
            recall_scores[k].append(compute_recall_at_k(retrieved_docids, relevant_docids, k))
            ndcg_scores[k].append(compute_ndcg_at_k(retrieved_docids, qrels[query_id], k))
        
        mrr_scores.append(compute_mrr(retrieved_docids, relevant_docids))
    
    # Aggregate metrics
    results = {
        "language": language,
        "split": split,
        "model": model_name,
        "num_queries": len(queries),
        "num_passages": len(all_passages),
        "metrics": {
            "MRR": float(np.mean(mrr_scores)),
        },
        "timing": {
            "query_encode_time": round(query_encode_time, 2),
            "passage_encode_time": round(passage_encode_time, 2),
            "total_encode_time": round(total_encode_time, 2),
            "query_speed": round(query_speed, 2),
            "passage_speed": round(passage_speed, 2),
        },
    }
    
    for k in top_k_values:
        results["metrics"][f"Recall@{k}"] = float(np.mean(recall_scores[k]))
        results["metrics"][f"nDCG@{k}"] = float(np.mean(ndcg_scores[k]))
    
    return results


def print_results(results: dict[str, Any]) -> None:
    """Pretty print evaluation results."""
    model_short = results['model'].split('/')[-1]
    print("\n" + "=" * 70)
    print(f"{model_short} Evaluation Results on MIRACL ({results['language'].upper()}, {results['split']})")
    print("=" * 70)
    print(f"Model: {results['model']}")
    print(f"Queries: {results['num_queries']}")
    print(f"Passages: {results['num_passages']}")
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
    
    # Print timing if available
    if "timing" in results:
        timing = results["timing"]
        print()
        print("Encoding Speed:")
        print(f"  Query encoding:   {timing['query_encode_time']:.2f}s ({timing['query_speed']:.2f} queries/sec)")
        print(f"  Passage encoding: {timing['passage_encode_time']:.2f}s ({timing['passage_speed']:.2f} passages/sec)")
        print(f"  Total time:       {timing['total_encode_time']:.2f}s")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 Embedding on MIRACL dataset"
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
        "--model", "-m",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Model name (default: Qwen/Qwen3-Embedding-0.6B)"
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
    parser.add_argument(
        "--query-instruction",
        default="",
        help="Instruction prefix for queries"
    )
    parser.add_argument(
        "--passage-instruction",
        default="",
        help="Instruction prefix for passages"
    )
    
    args = parser.parse_args()
    
    all_results = []
    
    for lang in args.language:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {args.model} on MIRACL ({lang.upper()}, {args.split})")
        logger.info(f"{'='*60}")
        
        try:
            results = evaluate_qwen3(
                language=lang,
                split=args.split,
                output_dir=args.output_dir,
                model_name=args.model,
                batch_size=args.batch_size,
                device=args.device,
                query_instruction=args.query_instruction,
                passage_instruction=args.passage_instruction,
            )
            all_results.append(results)
            print_results(results)
        except Exception as e:
            logger.error(f"Error evaluating {lang}: {e}")
            raise
    
    # Save results if requested
    if args.save_results and all_results:
        model_short = args.model.split('/')[-1].lower().replace('-', '_')
        output_path = Path(args.output_dir) / f"{model_short}_eval_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
