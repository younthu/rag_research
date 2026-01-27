"""Qwen3-VL-Embedding Evaluation on MIRACL Dataset.

This module provides evaluation of Qwen3-VL-Embedding model on the MIRACL benchmark.
It computes standard retrieval metrics including Recall@k, MRR, and nDCG.

Qwen3-VL-Embedding is a multimodal embedding model that handles text, images, and videos.
For text-only retrieval tasks like MIRACL, it can be used in text-only mode.

Usage:
    python -m miracl.qwen3_vl_eval --language en --split dev
    python -m miracl.qwen3_vl_eval --language zh --split dev
    python -m miracl.qwen3_vl_eval --language en zh --split dev  # Evaluate both
    python -m miracl.qwen3_vl_eval --model Qwen/Qwen3-VL-Embedding-8B  # Use larger model

Requirements:
    pip install transformers>=4.57.0 qwen-vl-utils>=0.0.14 torch
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Union
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Qwen3-VL-Embedding configuration
MAX_LENGTH = 8192
DEFAULT_INSTRUCTION = "Represent the user's input."


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


# ============================================================================
# Qwen3-VL-Embedding Model Classes (adapted from official implementation)
# ============================================================================

@dataclass
class Qwen3VLForEmbeddingOutput:
    """Output class for Qwen3-VL embedding model."""
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding:
    """Wrapper class for Qwen3-VL model to compute embeddings."""
    
    def __init__(self, model):
        self.model = model
        
    def __call__(self, **kwargs) -> Qwen3VLForEmbeddingOutput:
        outputs = self.model(**kwargs, output_hidden_states=True)
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.hidden_states[-1],
            attention_mask=kwargs.get('attention_mask'),
        )
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    @property
    def device(self):
        return next(self.model.parameters()).device


class Qwen3VLEmbedder:
    """Embedder class for Qwen3-VL-Embedding model.
    
    This class handles text-only inputs for retrieval tasks like MIRACL.
    For multimodal inputs (images, videos), additional processing is required.
    
    Note: For text-only tasks, we use AutoTokenizer instead of the full Processor
    to avoid the torchvision dependency required for video processing.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        default_instruction: str = DEFAULT_INSTRUCTION,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        from transformers import AutoTokenizer, AutoModel
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        self.max_length = max_length
        self.default_instruction = default_instruction
        
        # Set default dtype
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
        
        logger.info(f"Loading Qwen3-VL-Embedding model: {model_name_or_path}")
        logger.info(f"Device: {device}, dtype: {torch_dtype}")
        
        # Load model - Qwen3-VL-Embedding uses Qwen3VL architecture
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **kwargs
            )
        except ImportError:
            # Fallback to AutoModel if Qwen3VL not available
            model = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **kwargs
            )
        
        self.model = Qwen3VLForEmbedding(model).to(self.device).eval()
        
        # Load tokenizer only (not full processor) for text-only tasks
        # This avoids the torchvision dependency required by the video processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side='right',
            trust_remote_code=True,
        )
    
    def format_model_input(
        self,
        text: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        """Format text input into conversation format for Qwen3-VL.
        
        Args:
            text: Input text to embed.
            instruction: Custom instruction (uses default if not provided).
            
        Returns:
            Conversation in Qwen3-VL chat format.
        """
        import unicodedata
        
        # Ensure instruction ends with punctuation
        instruction = instruction or self.default_instruction
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
            instruction = instruction + '.'
        
        # Build conversation
        content = []
        if text:
            content.append({'type': 'text', 'text': text})
        else:
            content.append({'type': 'text', 'text': 'NULL'})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]
        
        return conversation
    
    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Preprocess conversations for model input (text-only)."""
        # Apply chat template to convert conversations to text
        texts = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors='pt',
        )
        
        return inputs
    
    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the last non-padding token's hidden state."""
        # Find the last non-padding position for each sequence
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        instruction: Optional[str] = None,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode.
            batch_size: Batch size for encoding.
            instruction: Custom instruction for all texts.
            normalize: Whether to L2-normalize embeddings.
            show_progress_bar: Whether to show progress bar.
            
        Returns:
            Numpy array of embeddings with shape (num_texts, embedding_dim).
        """
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        if show_progress_bar:
            try:
                from tqdm import tqdm
                batch_iterator = tqdm(range(num_batches), desc="Encoding")
            except ImportError:
                batch_iterator = range(num_batches)
        else:
            batch_iterator = range(num_batches)
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Format inputs
            conversations = [
                self.format_model_input(text=text, instruction=instruction)
                for text in batch_texts
            ]
            
            # Preprocess
            inputs = self._preprocess_inputs(conversations)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Pool embeddings
            embeddings = self._pooling_last(
                outputs.last_hidden_state,
                outputs.attention_mask
            )
            
            # Normalize if requested
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # Convert to float32 before numpy (numpy doesn't support bfloat16)
            all_embeddings.append(embeddings.float().cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)


def evaluate_qwen3_vl(
    language: str,
    split: str = "dev",
    output_dir: str = "output/miracl",
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    top_k_values: list[int] = [1, 5, 10, 20, 50, 100],
    batch_size: int = 8,
    max_length: int = 8192,
    device: str | None = None,
    query_instruction: str = "",
    passage_instruction: str = "",
) -> dict[str, Any]:
    """Evaluate Qwen3-VL-Embedding model on MIRACL dataset.
    
    Args:
        language: Language code (e.g., "en", "zh").
        split: Dataset split (e.g., "dev", "train").
        output_dir: Output directory for MIRACL data.
        model_name: Qwen3-VL-Embedding model name.
        top_k_values: List of k values for Recall@k and nDCG@k.
        batch_size: Batch size for encoding.
        max_length: Maximum token length.
        device: Device to use (e.g., "cuda", "cpu"). If None, auto-detect.
        query_instruction: Instruction prefix for queries.
        passage_instruction: Instruction prefix for passages.
        
    Returns:
        Dict containing evaluation metrics.
    """
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
                all_passages.append(passage_text)
    
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Total unique passages: {len(all_passages)}")
    
    # Load Qwen3-VL-Embedding model
    logger.info(f"Loading model: {model_name}...")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_name,
        max_length=max_length,
        device=device,
    )
    
    import time
    
    # Encode queries
    logger.info("Encoding queries...")
    query_start_time = time.time()
    query_embeddings = embedder.encode(
        queries,
        batch_size=batch_size,
        instruction=query_instruction if query_instruction else None,
        show_progress_bar=True,
    )
    query_encode_time = time.time() - query_start_time
    
    # Encode passages
    logger.info("Encoding passages...")
    passage_start_time = time.time()
    passage_embeddings = embedder.encode(
        all_passages,
        batch_size=batch_size,
        instruction=passage_instruction if passage_instruction else None,
        show_progress_bar=True,
    )
    passage_encode_time = time.time() - passage_start_time
    
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    logger.info(f"Passage embeddings shape: {passage_embeddings.shape}")
    
    # Calculate encoding speed
    query_speed = len(queries) / query_encode_time  # queries per second
    passage_speed = len(all_passages) / passage_encode_time  # passages per second
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


def load_qwen3_baseline_results(output_dir: str, language: str) -> dict[str, Any] | None:
    """Load Qwen3-Embedding baseline results for comparison.
    
    Searches for existing Qwen3-Embedding evaluation results in the output directory.
    """
    import glob
    
    output_path = Path(output_dir)
    
    # Search for Qwen3-Embedding result files
    patterns = [
        output_path / "qwen3_embedding*eval_results.json",
        output_path / "*qwen3*embedding*results.json",
    ]
    
    for pattern in patterns:
        for result_file in glob.glob(str(pattern)):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    # Handle both single result and list of results
                    if isinstance(results, list):
                        for r in results:
                            if r.get("language") == language and "Qwen3-Embedding" in r.get("model", ""):
                                logger.info(f"Found baseline results in {result_file}")
                                return r
                    elif results.get("language") == language:
                        return results
            except Exception as e:
                logger.debug(f"Could not load {result_file}: {e}")
    
    return None


def print_comparison(vl_results: dict[str, Any], baseline_results: dict[str, Any]) -> None:
    """Print a comparison table between Qwen3-VL-Embedding and Qwen3-Embedding."""
    vl_model = vl_results['model'].split('/')[-1]
    baseline_model = baseline_results['model'].split('/')[-1]
    lang = vl_results['language'].upper()
    split = vl_results['split']
    
    print("\n" + "=" * 85)
    print(f"Comparison: {vl_model} vs {baseline_model} on MIRACL ({lang}, {split})")
    print("=" * 85)
    
    vl_metrics = vl_results["metrics"]
    baseline_metrics = baseline_results["metrics"]
    
    # Header
    print(f"{'Metric':<15} | {baseline_model:<25} | {vl_model:<25} | {'Δ':>8}")
    print("-" * 85)
    
    # MRR
    vl_mrr = vl_metrics['MRR']
    bl_mrr = baseline_metrics['MRR']
    delta = vl_mrr - bl_mrr
    delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
    print(f"{'MRR':<15} | {bl_mrr:<25.4f} | {vl_mrr:<25.4f} | {delta_str:>8}")
    
    print("-" * 85)
    
    # Recall@k
    recall_keys = sorted([k for k in vl_metrics.keys() if k.startswith("Recall@")], 
                         key=lambda x: int(x.split("@")[1]))
    for key in recall_keys:
        if key in baseline_metrics:
            vl_val = vl_metrics[key]
            bl_val = baseline_metrics[key]
            delta = vl_val - bl_val
            delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
            print(f"{key:<15} | {bl_val:<25.4f} | {vl_val:<25.4f} | {delta_str:>8}")
    
    print("-" * 85)
    
    # nDCG@k
    ndcg_keys = sorted([k for k in vl_metrics.keys() if k.startswith("nDCG@")],
                       key=lambda x: int(x.split("@")[1]))
    for key in ndcg_keys:
        if key in baseline_metrics:
            vl_val = vl_metrics[key]
            bl_val = baseline_metrics[key]
            delta = vl_val - bl_val
            delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
            print(f"{key:<15} | {bl_val:<25.4f} | {vl_val:<25.4f} | {delta_str:>8}")
    
    # Timing comparison if available
    vl_timing = vl_results.get("timing")
    bl_timing = baseline_results.get("timing")
    
    if vl_timing:
        print("-" * 85)
        print("Encoding Speed:")
        
        if bl_timing:
            # Compare passage encoding speed
            vl_speed = vl_timing['passage_speed']
            bl_speed = bl_timing['passage_speed']
            speed_ratio = vl_speed / bl_speed if bl_speed > 0 else 0
            print(f"{'Passage Speed':<15} | {bl_speed:<22.2f}/sec | {vl_speed:<22.2f}/sec | {speed_ratio:.2f}x")
            
            # Compare total time
            vl_time = vl_timing['total_encode_time']
            bl_time = bl_timing['total_encode_time']
            time_ratio = bl_time / vl_time if vl_time > 0 else 0
            print(f"{'Total Time':<15} | {bl_time:<23.2f}s | {vl_time:<23.2f}s | {time_ratio:.2f}x")
        else:
            print(f"  VL Query:   {vl_timing['query_encode_time']:.2f}s ({vl_timing['query_speed']:.2f}/sec)")
            print(f"  VL Passage: {vl_timing['passage_encode_time']:.2f}s ({vl_timing['passage_speed']:.2f}/sec)")
            print(f"  (Baseline timing not available)")
    
    print("=" * 85)
    print("Note: Δ > 0 means Qwen3-VL-Embedding performs better on metrics")
    print("      Speed ratio > 1 means baseline is faster")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-Embedding on MIRACL dataset"
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
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Model name (default: Qwen/Qwen3-VL-Embedding-2B)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for encoding (default: 8, smaller due to VL model size)"
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
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline results JSON file (e.g., qwen3_embedding_0.6b_eval_results.json) for comparison"
    )
    
    args = parser.parse_args()
    
    all_results = []
    
    for lang in args.language:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {args.model} on MIRACL ({lang.upper()}, {args.split})")
        logger.info(f"{'='*60}")
        
        try:
            results = evaluate_qwen3_vl(
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
            
            # Try to load and compare with Qwen3-Embedding baseline
            baseline = None
            if args.baseline:
                # Load baseline from specified file
                try:
                    with open(args.baseline, "r", encoding="utf-8") as f:
                        baseline_data = json.load(f)
                        if isinstance(baseline_data, list):
                            for r in baseline_data:
                                if r.get("language") == lang:
                                    baseline = r
                                    break
                        else:
                            baseline = baseline_data
                except Exception as e:
                    logger.warning(f"Could not load baseline file: {e}")
            else:
                baseline = load_qwen3_baseline_results(args.output_dir, lang)
            
            if baseline:
                print_comparison(results, baseline)
            else:
                logger.info(f"No Qwen3-Embedding baseline found for {lang}. "
                           "Use --baseline to specify a results file for comparison.")
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
