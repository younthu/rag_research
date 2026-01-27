"""HotpotQA dataset assets for RAG evaluation.

This module provides Dagster assets for:
1. Downloading HotpotQA dataset from HuggingFace
2. Processing multi-hop QA data with supporting facts
3. Building RAG evaluation dataset with relevance judgments

HotpotQA is a multi-hop question answering dataset where each question requires
reasoning over 2 supporting paragraphs. The distractor setting includes 10 total
paragraphs per question (2 gold + 8 distractors).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dagster import asset, AssetExecutionContext, Config
from huggingface_hub import hf_hub_download, list_repo_files


def _convert_numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy arrays and types to Python native types.
    
    Args:
        obj: Object to convert.
        
    Returns:
        Object with numpy types converted to Python types.
    """
    if isinstance(obj, np.ndarray):
        return [_convert_numpy_to_python(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_python(item) for item in obj]
    return obj


class HotpotQAConfig(Config):
    """Configuration for HotpotQA dataset assets."""

    subset: str = "distractor"  # distractor or fullwiki
    split: str = "validation"  # train or validation
    output_dir: str = "output/hotpotqa"


def _download_hotpotqa(
    subset: str, split: str, context: AssetExecutionContext
) -> list[dict]:
    """Download HotpotQA dataset from HuggingFace using parquet files.
    
    Args:
        subset: Dataset subset ('distractor' or 'fullwiki').
        split: Dataset split ('train' or 'validation').
        context: Dagster execution context for logging.
        
    Returns:
        List of dataset items.
    """
    repo_id = "hotpot_qa"
    
    context.log.info(f"Listing files in {repo_id}...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    # Find parquet files for the target subset and split
    # HotpotQA structure: {subset}/{split}-*.parquet
    parquet_files = []
    for f in all_files:
        if f.endswith(".parquet"):
            # Check for pattern like: distractor/validation-00000-of-00001.parquet
            if f.startswith(f"{subset}/") and f"/{split}-" in f:
                parquet_files.append(f)
            # Also check for refs/convert/parquet pattern
            elif subset in f and split in f:
                parquet_files.append(f)
    
    context.log.info(f"Found {len(parquet_files)} parquet files: {parquet_files}")
    
    if not parquet_files:
        # Try refs/convert/parquet branch
        context.log.info("Trying refs/convert/parquet branch...")
        try:
            all_files = list_repo_files(repo_id, repo_type="dataset", revision="refs/convert/parquet")
            for f in all_files:
                if f.endswith(".parquet") and subset in f and split in f:
                    parquet_files.append(f)
            context.log.info(f"Found {len(parquet_files)} parquet files in convert branch")
        except Exception as e:
            context.log.warning(f"Failed to list convert branch: {e}")
    
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for {subset}/{split}. "
            f"Available files: {[f for f in all_files if f.endswith('.parquet')]}"
        )
    
    # Download and load parquet files
    items = []
    for pf in parquet_files:
        context.log.info(f"Downloading {pf}...")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pf,
                repo_type="dataset",
            )
        except Exception:
            # Try with convert branch
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pf,
                repo_type="dataset",
                revision="refs/convert/parquet",
            )
        
        df = pd.read_parquet(local_path)
        items.extend(df.to_dict("records"))
    
    context.log.info(f"Loaded {len(items)} items from parquet files")
    return items


@asset(
    description="Download HotpotQA dataset from HuggingFace datasets",
    group_name="hotpotqa",
)
def hotpotqa_download(context: AssetExecutionContext, config: HotpotQAConfig) -> dict[str, Any]:
    """Download and cache the HotpotQA dataset.
    
    The dataset contains multi-hop questions with:
    - question: The question text
    - answer: The answer text
    - supporting_facts: List of [title, sentence_idx] for gold paragraphs
    - context: List of [title, sentences] with 2 gold + 8 distractor paragraphs
    - type: Question type (comparison or bridge)
    - level: Difficulty level (easy, medium, hard)
    
    Returns:
        Dict containing dataset metadata and paths.
    """
    context.log.info(f"Downloading HotpotQA dataset: subset={config.subset}, split={config.split}")
    
    # Download the dataset using datasets library
    items = _download_hotpotqa(config.subset, config.split, context)
    
    context.log.info(f"Loaded {len(items)} questions")
    
    # Save to JSONL
    output_path = Path(config.output_dir) / config.subset / "raw"
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_file = output_path / f"{config.split}.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for item in items:
            # Convert numpy arrays to Python lists for JSON serialization
            item_converted = _convert_numpy_to_python(item)
            f.write(json.dumps(item_converted, ensure_ascii=False) + "\n")
    
    context.log.info(f"Saved {len(items)} items to {data_file}")
    
    # Compute statistics
    type_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    
    for item in items:
        qtype = item.get("type", "unknown")
        level = item.get("level", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        level_counts[level] = level_counts.get(level, 0) + 1
    
    context.log.info(f"Question types: {type_counts}")
    context.log.info(f"Difficulty levels: {level_counts}")
    
    return {
        "subset": config.subset,
        "split": config.split,
        "question_count": len(items),
        "type_counts": type_counts,
        "level_counts": level_counts,
        "data_path": str(data_file),
    }


@asset(
    deps=[hotpotqa_download],
    description="Process HotpotQA data and extract passages",
    group_name="hotpotqa",
)
def hotpotqa_passages(context: AssetExecutionContext, config: HotpotQAConfig) -> dict[str, Any]:
    """Extract and process passages from HotpotQA dataset.
    
    For each question, extracts:
    - Supporting paragraphs (gold passages)
    - Distractor paragraphs
    
    Returns:
        Dict containing passage metadata and paths.
    """
    data_path = Path(config.output_dir) / config.subset / "raw" / f"{config.split}.jsonl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run hotpotqa_download first.")
    
    context.log.info(f"Processing passages from {data_path}")
    
    # Load data
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    # Extract all unique passages
    all_passages: dict[str, dict] = {}  # title -> passage info
    
    for item in items:
        context_data = item.get("context", {})
        
        # Handle different context formats
        if isinstance(context_data, dict):
            # Format: {"title": [...], "sentences": [...]}
            titles = context_data.get("title", [])
            sentences_list = context_data.get("sentences", [])
            
            for i, title in enumerate(titles):
                if title not in all_passages:
                    sentences = sentences_list[i] if i < len(sentences_list) else []
                    if isinstance(sentences, list):
                        text = " ".join(sentences)
                    else:
                        text = str(sentences)
                    
                    all_passages[title] = {
                        "title": title,
                        "text": text,
                        "sentences": sentences if isinstance(sentences, list) else [sentences],
                    }
        elif isinstance(context_data, list):
            # Format: [[title, sentences], ...]
            for ctx_item in context_data:
                if isinstance(ctx_item, (list, tuple)) and len(ctx_item) >= 2:
                    title = ctx_item[0]
                    sentences = ctx_item[1]
                    
                    if title not in all_passages:
                        if isinstance(sentences, list):
                            text = " ".join(sentences)
                        else:
                            text = str(sentences)
                        
                        all_passages[title] = {
                            "title": title,
                            "text": text,
                            "sentences": sentences if isinstance(sentences, list) else [sentences],
                        }
    
    context.log.info(f"Extracted {len(all_passages)} unique passages")
    
    # Save passages
    output_path = Path(config.output_dir) / config.subset / "passages"
    output_path.mkdir(parents=True, exist_ok=True)
    
    passages_file = output_path / f"passages_{config.split}.jsonl"
    with open(passages_file, "w", encoding="utf-8") as f:
        for passage in all_passages.values():
            f.write(json.dumps(passage, ensure_ascii=False) + "\n")
    
    # Save passage index (title -> id mapping)
    passage_index = {title: idx for idx, title in enumerate(all_passages.keys())}
    index_file = output_path / f"passage_index_{config.split}.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(passage_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Saved passages to {passages_file}")
    context.log.info(f"Saved passage index to {index_file}")
    
    return {
        "subset": config.subset,
        "split": config.split,
        "passage_count": len(all_passages),
        "passages_path": str(passages_file),
        "index_path": str(index_file),
    }


@asset(
    deps=[hotpotqa_passages],
    description="Convert HotpotQA passages to markdown files",
    group_name="hotpotqa",
)
def hotpotqa_markdown_docs(context: AssetExecutionContext, config: HotpotQAConfig) -> dict[str, Any]:
    """Convert HotpotQA passages to individual markdown files.
    
    Each passage is saved as a markdown file with:
    - Title as H1 header
    - Passage text as content
    
    Returns:
        Dict containing markdown output metadata.
    """
    passages_path = Path(config.output_dir) / config.subset / "passages" / f"passages_{config.split}.jsonl"
    
    if not passages_path.exists():
        raise FileNotFoundError(f"Passages file not found: {passages_path}. Run hotpotqa_passages first.")
    
    md_output_dir = Path(config.output_dir) / config.subset / "markdown_docs" / config.split
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting passages to markdown in {md_output_dir}")
    
    doc_count = 0
    doc_index = {}
    used_filenames: set[str] = set()
    
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            passage = json.loads(line)
            
            title = passage["title"]
            text = passage["text"]
            
            # Create markdown content
            md_content = f"# {title}\n\n{text}\n"
            
            # Sanitize title for filename
            safe_title = title.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_title = safe_title.replace("*", "_").replace("?", "_").replace('"', "_")
            safe_title = safe_title.replace("<", "_").replace(">", "_").replace("|", "_")
            safe_title = safe_title.strip()[:100]  # Limit length
            
            if not safe_title:
                safe_title = f"passage_{doc_count}"
            
            # Handle filename collision
            if safe_title in used_filenames:
                safe_title = f"{safe_title}_{doc_count}"
            
            used_filenames.add(safe_title)
            md_file = md_output_dir / f"{safe_title}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[title] = {
                "path": str(md_file),
                "title": title,
            }
            doc_count += 1
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} passages to markdown")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "subset": config.subset,
        "split": config.split,
        "document_count": doc_count,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[hotpotqa_passages],
    description="Build HotpotQA evaluation dataset for RAG retrieval",
    group_name="hotpotqa",
)
def hotpotqa_eval_dataset(context: AssetExecutionContext, config: HotpotQAConfig) -> dict[str, Any]:
    """Build evaluation dataset for RAG retrieval evaluation.
    
    For each question, creates an entry with:
    - query_id: Unique question identifier
    - query: Question text
    - answer: Answer text
    - type: Question type (comparison or bridge)
    - level: Difficulty level (easy, medium, hard)
    - relevant_passages: List of gold supporting paragraphs
    - distractor_passages: List of distractor paragraphs
    
    Returns:
        Dict containing evaluation dataset metadata.
    """
    data_path = Path(config.output_dir) / config.subset / "raw" / f"{config.split}.jsonl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run hotpotqa_download first.")
    
    context.log.info(f"Building evaluation dataset from {data_path}")
    
    # Load data
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    eval_data = []
    
    for idx, item in enumerate(items):
        question = item.get("question", "")
        answer = item.get("answer", "")
        qtype = item.get("type", "unknown")
        level = item.get("level", "unknown")
        
        # Get supporting facts (gold passages)
        supporting_facts = item.get("supporting_facts", {})
        if isinstance(supporting_facts, dict):
            gold_titles = set(supporting_facts.get("title", []))
        elif isinstance(supporting_facts, list):
            gold_titles = set(sf[0] for sf in supporting_facts if isinstance(sf, (list, tuple)))
        else:
            gold_titles = set()
        
        # Process context to get all passages
        context_data = item.get("context", {})
        all_passages: dict[str, dict] = {}
        
        if isinstance(context_data, dict):
            titles = context_data.get("title", [])
            sentences_list = context_data.get("sentences", [])
            
            for i, title in enumerate(titles):
                sentences = sentences_list[i] if i < len(sentences_list) else []
                if isinstance(sentences, list):
                    text = " ".join(sentences)
                else:
                    text = str(sentences)
                
                all_passages[title] = {
                    "passage_id": f"{idx}_{title}",
                    "title": title,
                    "text": text,
                }
        elif isinstance(context_data, list):
            for ctx_item in context_data:
                if isinstance(ctx_item, (list, tuple)) and len(ctx_item) >= 2:
                    title = ctx_item[0]
                    sentences = ctx_item[1]
                    
                    if isinstance(sentences, list):
                        text = " ".join(sentences)
                    else:
                        text = str(sentences)
                    
                    all_passages[title] = {
                        "passage_id": f"{idx}_{title}",
                        "title": title,
                        "text": text,
                    }
        
        # Split into relevant and distractor passages
        relevant_passages = []
        distractor_passages = []
        
        for title, passage in all_passages.items():
            if title in gold_titles:
                relevant_passages.append(passage)
            else:
                distractor_passages.append(passage)
        
        eval_entry = {
            "query_id": item.get("id", f"q_{idx}"),
            "query": question,
            "answer": answer,
            "type": qtype,
            "level": level,
            "relevant_passages": relevant_passages,
            "distractor_passages": distractor_passages,
        }
        eval_data.append(eval_entry)
    
    context.log.info(f"Built evaluation dataset with {len(eval_data)} entries")
    
    # Save evaluation dataset
    eval_output_dir = Path(config.output_dir) / config.subset / "eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_file = eval_output_dir / f"eval_dataset_{config.split}.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for entry in eval_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Compute summary statistics
    total_relevant = sum(len(e["relevant_passages"]) for e in eval_data)
    total_distractor = sum(len(e["distractor_passages"]) for e in eval_data)
    
    type_counts = {}
    level_counts = {}
    for entry in eval_data:
        qtype = entry["type"]
        level = entry["level"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        level_counts[level] = level_counts.get(level, 0) + 1
    
    summary = {
        "subset": config.subset,
        "split": config.split,
        "total_queries": len(eval_data),
        "total_relevant_passages": total_relevant,
        "total_distractor_passages": total_distractor,
        "avg_relevant_per_query": total_relevant / len(eval_data) if eval_data else 0,
        "avg_distractor_per_query": total_distractor / len(eval_data) if eval_data else 0,
        "type_counts": type_counts,
        "level_counts": level_counts,
        "eval_path": str(eval_file),
    }
    
    # Save summary
    summary_file = eval_output_dir / f"eval_summary_{config.split}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Saved evaluation dataset to {eval_file}")
    context.log.info(f"Saved summary to {summary_file}")
    context.log.info(f"Average relevant passages per query: {summary['avg_relevant_per_query']:.2f}")
    context.log.info(f"Average distractor passages per query: {summary['avg_distractor_per_query']:.2f}")
    
    return summary


@asset(
    deps=[hotpotqa_download],
    description="Extract supporting facts from HotpotQA for sentence-level retrieval",
    group_name="hotpotqa",
)
def hotpotqa_supporting_facts(context: AssetExecutionContext, config: HotpotQAConfig) -> dict[str, Any]:
    """Extract supporting facts at sentence level for fine-grained evaluation.
    
    Supporting facts are the specific sentences within passages that contain
    the evidence needed to answer the question.
    
    Returns:
        Dict containing supporting facts metadata.
    """
    data_path = Path(config.output_dir) / config.subset / "raw" / f"{config.split}.jsonl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run hotpotqa_download first.")
    
    context.log.info(f"Extracting supporting facts from {data_path}")
    
    # Load data
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    facts_data = []
    
    for idx, item in enumerate(items):
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # Get supporting facts
        supporting_facts = item.get("supporting_facts", {})
        
        if isinstance(supporting_facts, dict):
            titles = supporting_facts.get("title", [])
            sent_ids = supporting_facts.get("sent_id", [])
            facts = list(zip(titles, sent_ids))
        elif isinstance(supporting_facts, list):
            facts = [(sf[0], sf[1]) for sf in supporting_facts if isinstance(sf, (list, tuple)) and len(sf) >= 2]
        else:
            facts = []
        
        # Get context for sentence lookup
        context_data = item.get("context", {})
        title_to_sentences: dict[str, list[str]] = {}
        
        if isinstance(context_data, dict):
            titles = context_data.get("title", [])
            sentences_list = context_data.get("sentences", [])
            
            for i, title in enumerate(titles):
                sentences = sentences_list[i] if i < len(sentences_list) else []
                if isinstance(sentences, list):
                    title_to_sentences[title] = sentences
                else:
                    title_to_sentences[title] = [str(sentences)]
        elif isinstance(context_data, list):
            for ctx_item in context_data:
                if isinstance(ctx_item, (list, tuple)) and len(ctx_item) >= 2:
                    title = ctx_item[0]
                    sentences = ctx_item[1]
                    if isinstance(sentences, list):
                        title_to_sentences[title] = sentences
                    else:
                        title_to_sentences[title] = [str(sentences)]
        
        # Extract supporting sentences
        supporting_sentences = []
        for title, sent_id in facts:
            sentences = title_to_sentences.get(title, [])
            if isinstance(sent_id, int) and sent_id < len(sentences):
                supporting_sentences.append({
                    "title": title,
                    "sentence_id": sent_id,
                    "sentence": sentences[sent_id],
                })
        
        facts_entry = {
            "query_id": item.get("id", f"q_{idx}"),
            "question": question,
            "answer": answer,
            "supporting_facts": facts,
            "supporting_sentences": supporting_sentences,
            "num_facts": len(facts),
        }
        facts_data.append(facts_entry)
    
    context.log.info(f"Extracted supporting facts for {len(facts_data)} questions")
    
    # Save supporting facts
    output_path = Path(config.output_dir) / config.subset / "facts"
    output_path.mkdir(parents=True, exist_ok=True)
    
    facts_file = output_path / f"supporting_facts_{config.split}.jsonl"
    with open(facts_file, "w", encoding="utf-8") as f:
        for entry in facts_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Compute statistics
    total_facts = sum(e["num_facts"] for e in facts_data)
    avg_facts = total_facts / len(facts_data) if facts_data else 0
    
    context.log.info(f"Total supporting facts: {total_facts}")
    context.log.info(f"Average facts per question: {avg_facts:.2f}")
    context.log.info(f"Saved to {facts_file}")
    
    return {
        "subset": config.subset,
        "split": config.split,
        "question_count": len(facts_data),
        "total_supporting_facts": total_facts,
        "avg_facts_per_question": avg_facts,
        "facts_path": str(facts_file),
    }
