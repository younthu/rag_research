"""MIRACL dataset assets for RAG evaluation.

This module provides Dagster assets for:
1. Downloading MIRACL dataset (corpus and qrels)
2. Converting documents to markdown files
3. Building RAG evaluation dataset with relevance judgments
"""

import email
import json
import tarfile
import urllib.request
from email import policy
from pathlib import Path
from typing import Any

import pandas as pd
from dagster import asset, AssetExecutionContext, Config
from huggingface_hub import hf_hub_download, list_repo_files


class MiraclConfig(Config):
    """Configuration for MIRACL dataset assets."""

    language: str = "zh"  # MIRACL supports: ar, bn, de, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, vi, yo, zh
    split: str = "dev"  # train or dev
    output_dir: str = "output/miracl"


def _download_parquet_files(repo_id: str, config_name: str, split: str, cache_dir: Path) -> list[Path]:
    """Download parquet files from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "miracl/miracl-corpus")
        config_name: Dataset configuration name (e.g., "zh")
        split: Dataset split (e.g., "train", "dev")
        cache_dir: Local cache directory
        
    Returns:
        List of downloaded parquet file paths.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # List all files in the repo
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    # Filter for parquet files matching the config and split
    # Pattern: data/{config}/{split}-*.parquet or {config}/{split}-*.parquet
    parquet_files = []
    for f in all_files:
        if f.endswith(".parquet"):
            # Check if file matches our config and split
            if f"/{config_name}/" in f or f.startswith(f"{config_name}/"):
                if f"/{split}-" in f or f"/{split}/" in f or f"-{split}-" in f:
                    parquet_files.append(f)
            # Also check for pattern like: miracl-corpus-v1.0-zh/train-00000-of-00001.parquet
            elif f"-{config_name}/" in f:
                if f"/{split}-" in f:
                    parquet_files.append(f)
    
    # If no parquet files found, try the refs/convert/parquet branch
    if not parquet_files:
        try:
            all_files = list_repo_files(repo_id, repo_type="dataset", revision="refs/convert/parquet")
            for f in all_files:
                if f.endswith(".parquet"):
                    if config_name in f and split in f:
                        parquet_files.append(f)
        except Exception:
            pass
    
    downloaded_files = []
    for pf in parquet_files:
        local_path = cache_dir / Path(pf).name
        if not local_path.exists():
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=pf,
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
                downloaded_files.append(Path(downloaded))
            except Exception:
                # Try with convert/parquet revision
                try:
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=pf,
                        repo_type="dataset",
                        revision="refs/convert/parquet",
                        local_dir=cache_dir,
                    )
                    downloaded_files.append(Path(downloaded))
                except Exception:
                    pass
        else:
            downloaded_files.append(local_path)
    
    return downloaded_files


@asset(
    description="Download MIRACL corpus from HuggingFace datasets",
    group_name="miracl",
)
def miracl_corpus(context: AssetExecutionContext, config: MiraclConfig) -> dict[str, Any]:
    """Download and cache the MIRACL document corpus.
    
    The corpus contains documents with:
    - docid: Unique document identifier
    - title: Document title
    - text: Document content
    
    Returns:
        Dict containing corpus metadata and document count.
    """
    context.log.info(f"Downloading MIRACL corpus for language: {config.language}")
    
    # Try direct parquet download first
    cache_dir = Path(config.output_dir) / config.language / "cache" / "corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download parquet files from HuggingFace Hub
    repo_id = "miracl/miracl-corpus"
    
    context.log.info("Listing files in miracl/miracl-corpus...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    # Find parquet files for the target language
    # The corpus typically has pattern: miracl-corpus-v1.0-{lang}/*.parquet
    parquet_files = []
    for f in all_files:
        if f.endswith(".parquet") and config.language in f:
            parquet_files.append(f)
    
    context.log.info(f"Found {len(parquet_files)} parquet files for {config.language}")
    
    if not parquet_files:
        # Fallback: try using datasets with trust_remote_code (older datasets versions)
        context.log.warning("No parquet files found, trying legacy load_dataset...")
        from datasets import load_dataset
        
        corpus = load_dataset(
            "miracl/miracl-corpus",
            config.language,
            split="train",
            trust_remote_code=True,
        )
        docs = list(corpus)
    else:
        # Download and load parquet files
        docs = []
        for pf in parquet_files:
            context.log.info(f"Downloading {pf}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pf,
                repo_type="dataset",
            )
            df = pd.read_parquet(local_path)
            docs.extend(df.to_dict("records"))
        
        context.log.info(f"Loaded {len(docs)} documents from parquet files")
    
    # Save corpus to local cache
    output_path = Path(config.output_dir) / config.language / "corpus"
    output_path.mkdir(parents=True, exist_ok=True)
    
    corpus_file = output_path / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    context.log.info(f"Saved corpus to {corpus_file}")
    
    return {
        "language": config.language,
        "document_count": len(docs),
        "corpus_path": str(corpus_file),
    }


@asset(
    description="Download MIRACL topics and qrels from HuggingFace datasets",
    group_name="miracl",
)
def miracl_qrels(context: AssetExecutionContext, config: MiraclConfig) -> dict[str, Any]:
    """Download topics (queries) and relevance judgments (qrels).
    
    Topics contain:
    - query_id: Unique query identifier
    - query: Query text
    
    Qrels contain relevance judgments:
    - query_id: Query identifier
    - docid: Document identifier
    - relevance: Relevance score (0 = not relevant, 1 = relevant)
    
    Returns:
        Dict containing qrels metadata and counts.
    """
    context.log.info(f"Downloading MIRACL qrels for language: {config.language}, split: {config.split}")
    
    repo_id = "miracl/miracl"
    
    context.log.info("Listing files in miracl/miracl...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    # Find parquet files for the target language and split
    parquet_files = []
    for f in all_files:
        if f.endswith(".parquet") and config.language in f and config.split in f:
            parquet_files.append(f)
    
    context.log.info(f"Found {len(parquet_files)} parquet files for {config.language}/{config.split}")
    
    if not parquet_files:
        # Fallback: try using datasets with trust_remote_code
        context.log.warning("No parquet files found, trying legacy load_dataset...")
        from datasets import load_dataset
        
        dataset = load_dataset(
            "miracl/miracl",
            config.language,
            split=config.split,
            trust_remote_code=True,
        )
        items = list(dataset)
    else:
        # Download and load parquet files
        items = []
        for pf in parquet_files:
            context.log.info(f"Downloading {pf}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pf,
                repo_type="dataset",
            )
            df = pd.read_parquet(local_path)
            items.extend(df.to_dict("records"))
        
        context.log.info(f"Loaded {len(items)} queries from parquet files")
    
    context.log.info(f"Loaded {len(items)} queries with relevance judgments")
    
    # Save to local cache
    output_path = Path(config.output_dir) / config.language / "qrels"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save topics (queries)
    topics_file = output_path / f"topics_{config.split}.jsonl"
    qrels_file = output_path / f"qrels_{config.split}.jsonl"
    
    total_judgments = 0
    with open(topics_file, "w", encoding="utf-8") as tf, \
         open(qrels_file, "w", encoding="utf-8") as qf:
        for item in items:
            # Write topic
            topic = {
                "query_id": item["query_id"],
                "query": item["query"],
            }
            tf.write(json.dumps(topic, ensure_ascii=False) + "\n")
            
            # Write qrels (relevance judgments)
            # positive_passages have relevance = 1
            positive_passages = item.get("positive_passages") or []
            if isinstance(positive_passages, dict):
                # Handle case where passages are stored as dict with lists
                docids = positive_passages.get("docid", [])
                titles = positive_passages.get("title", [])
                texts = positive_passages.get("text", [])
                for i in range(len(docids)):
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": docids[i] if i < len(docids) else "",
                        "relevance": 1,
                        "title": titles[i] if i < len(titles) else "",
                        "text": texts[i] if i < len(texts) else "",
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
            else:
                for passage in positive_passages:
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": passage["docid"],
                        "relevance": 1,
                        "title": passage.get("title", ""),
                        "text": passage.get("text", ""),
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
            
            # negative_passages have relevance = 0
            negative_passages = item.get("negative_passages") or []
            if isinstance(negative_passages, dict):
                docids = negative_passages.get("docid", [])
                titles = negative_passages.get("title", [])
                texts = negative_passages.get("text", [])
                for i in range(len(docids)):
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": docids[i] if i < len(docids) else "",
                        "relevance": 0,
                        "title": titles[i] if i < len(titles) else "",
                        "text": texts[i] if i < len(texts) else "",
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
            else:
                for passage in negative_passages:
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": passage["docid"],
                        "relevance": 0,
                        "title": passage.get("title", ""),
                        "text": passage.get("text", ""),
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
    
    context.log.info(f"Saved topics to {topics_file}")
    context.log.info(f"Saved qrels to {qrels_file}")
    
    return {
        "language": config.language,
        "split": config.split,
        "query_count": len(items),
        "judgment_count": total_judgments,
        "topics_path": str(topics_file),
        "qrels_path": str(qrels_file),
    }


@asset(
    deps=[miracl_qrels],
    description="Extract unique base docids from MIRACL qrels",
    group_name="miracl",
)
def miracl_unique_docids(context: AssetExecutionContext, config: MiraclConfig) -> dict[str, Any]:
    """Extract all unique base docids from qrels file.
    
    The docid format in qrels is: {base_docid}#{paragraph_id}
    This asset extracts unique base docids (the part before #).
    
    For example:
    - "6007#26" -> base_docid is "6007"
    - "14224#35" -> base_docid is "14224"
    
    Returns:
        Dict containing unique docid metadata and file path.
    """
    qrels_path = Path(config.output_dir) / config.language / "qrels" / f"qrels_{config.split}.jsonl"
    
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}. Run miracl_qrels first.")
    
    context.log.info(f"Extracting unique docids from {qrels_path}")
    
    unique_docids: set[str] = set()
    total_entries = 0
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            qrel = json.loads(line)
            docid = qrel["docid"]
            
            # Extract base_docid (part before #)
            if "#" in docid:
                base_docid = docid.split("#")[0]
            else:
                base_docid = docid
            
            unique_docids.add(base_docid)
            total_entries += 1
    
    context.log.info(f"Processed {total_entries} qrel entries")
    context.log.info(f"Found {len(unique_docids)} unique base docids")
    
    # Save unique docids to file
    output_path = Path(config.output_dir) / config.language / "qrels"
    output_path.mkdir(parents=True, exist_ok=True)
    
    docids_file = output_path / f"unique_docids_{config.split}.json"
    sorted_docids = sorted(unique_docids, key=lambda x: int(x) if x.isdigit() else x)
    
    with open(docids_file, "w", encoding="utf-8") as f:
        json.dump({
            "unique_docid_count": len(unique_docids),
            "docids": sorted_docids,
        }, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Saved unique docids to {docids_file}")
    
    return {
        "language": config.language,
        "split": config.split,
        "total_qrel_entries": total_entries,
        "unique_docid_count": len(unique_docids),
        "docids_path": str(docids_file),
    }


@asset(
    deps=[miracl_unique_docids, miracl_corpus],
    description="Extract corpus documents that match unique docids from qrels and save as markdown",
    group_name="miracl",
)
def miracl_filtered_corpus(context: AssetExecutionContext, config: MiraclConfig) -> dict[str, Any]:
    """Extract only documents from corpus that have matching docids in qrels.
    
    This asset filters the full corpus to include only documents whose base_docid
    appears in the unique docids extracted from qrels. Each document is saved as
    a markdown file with:
    - Title as H1 header
    - Paragraphs joined by two newlines
    
    Returns:
        Dict containing filtered corpus metadata and output directory path.
    """
    # Load unique docids
    docids_path = Path(config.output_dir) / config.language / "qrels" / f"unique_docids_{config.split}.json"
    
    if not docids_path.exists():
        raise FileNotFoundError(f"Unique docids file not found: {docids_path}. Run miracl_unique_docids first.")
    
    with open(docids_path, "r", encoding="utf-8") as f:
        docids_data = json.load(f)
    
    unique_docids = set(docids_data["docids"])
    context.log.info(f"Loaded {len(unique_docids)} unique docids to filter")
    
    # Load and filter corpus
    corpus_path = Path(config.output_dir) / config.language / "corpus" / "corpus.jsonl"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run miracl_corpus first.")
    
    context.log.info(f"Filtering corpus from {corpus_path}")
    
    # First pass: collect paragraphs for matching documents
    # Format: {base_docid: {"title": str, "paragraphs": [(paragraph_id, text), ...]}}
    docs_by_id: dict[str, dict] = {}
    total_paragraphs = 0
    filtered_paragraphs = 0
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            total_paragraphs += 1
            doc = json.loads(line)
            docid = doc["docid"]
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            # Parse docid: base_docid#paragraph_id
            if "#" in docid:
                base_docid, para_id = docid.rsplit("#", 1)
                try:
                    para_id = int(para_id)
                except ValueError:
                    para_id = 0
            else:
                base_docid = docid
                para_id = 0
            
            # Only keep documents whose base_docid is in unique_docids
            if base_docid in unique_docids:
                if base_docid not in docs_by_id:
                    docs_by_id[base_docid] = {"title": title, "paragraphs": []}
                
                docs_by_id[base_docid]["paragraphs"].append((para_id, text))
                filtered_paragraphs += 1
            
            if total_paragraphs % 100000 == 0:
                context.log.info(f"Processed {total_paragraphs} paragraphs, kept {filtered_paragraphs}...")
    
    context.log.info(f"Filtered {filtered_paragraphs} paragraphs from {total_paragraphs} total")
    context.log.info(f"Found {len(docs_by_id)} unique documents to write")
    
    # Output path for filtered markdown files
    md_output_dir = Path(config.output_dir) / config.language / "filtered_corpus" / config.split
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write each document as a markdown file
    doc_count = 0
    doc_index = {}  # Map base_docid to markdown file path
    used_filenames: set[str] = set()  # Track used filenames to avoid collision
    
    for base_docid, doc_data in docs_by_id.items():
        title = doc_data["title"]
        paragraphs = doc_data["paragraphs"]
        
        # Sort paragraphs by paragraph_id
        paragraphs.sort(key=lambda x: x[0])
        
        # Join paragraphs with two newlines
        paragraph_texts = [p[1] for p in paragraphs]
        combined_text = "\n\n".join(paragraph_texts)
        
        # Create markdown content: title as H1, then all paragraphs
        md_content = f"# {title}\n\n{combined_text}\n"
        
        # Use title as filename (sanitize for filesystem)
        # Remove or replace characters that are invalid in filenames
        safe_title = title.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe_title = safe_title.replace("*", "_").replace("?", "_").replace('"', "_")
        safe_title = safe_title.replace("<", "_").replace(">", "_").replace("|", "_")
        safe_title = safe_title.strip()
        
        # Handle empty title
        if not safe_title:
            safe_title = f"untitled_{base_docid}"
        
        # Handle filename collision by appending docid
        if safe_title in used_filenames:
            safe_title = f"{safe_title}_{base_docid}"
        
        used_filenames.add(safe_title)
        md_file = md_output_dir / f"{safe_title}.md"
        
        with open(md_file, "w", encoding="utf-8") as mf:
            mf.write(md_content)
        
        doc_index[base_docid] = str(md_file)
        doc_count += 1
        
        if doc_count % 1000 == 0:
            context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Wrote {doc_count} markdown files to {md_output_dir}")
    context.log.info(f"Document index saved to {index_file}")
    
    # Check for missing docids
    matched_docids = set(docs_by_id.keys())
    missing_docids = unique_docids - matched_docids
    if missing_docids:
        context.log.warning(f"Warning: {len(missing_docids)} docids from qrels not found in corpus")
        # Save missing docids for reference
        missing_file = md_output_dir / "missing_docids.json"
        with open(missing_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(missing_docids)), f, ensure_ascii=False, indent=2)
        context.log.info(f"Missing docids saved to {missing_file}")
    
    return {
        "language": config.language,
        "split": config.split,
        "total_corpus_paragraphs": total_paragraphs,
        "filtered_paragraphs": filtered_paragraphs,
        "document_count": doc_count,
        "matched_docid_count": len(matched_docids),
        "missing_docid_count": len(missing_docids),
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[miracl_corpus],
    description="Convert MIRACL corpus documents to markdown files",
    group_name="miracl",
)
def miracl_markdown_docs(context: AssetExecutionContext, config: MiraclConfig) -> dict[str, Any]:
    """Convert MIRACL documents to individual markdown files.
    
    Documents in corpus have docid format: {base_docid}#{paragraph_id}
    This function groups paragraphs by base_docid and concatenates them into
    a single markdown file per document.
    
    Each document is saved as a separate markdown file with:
    - Filename: {base_docid}.md
    - Content: Title as H1, followed by all paragraphs separated by blank lines
    
    Returns:
        Dict containing markdown output metadata.
    """
    # Read corpus from cached file
    corpus_path = Path(config.output_dir) / config.language / "corpus" / "corpus.jsonl"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run miracl_corpus first.")
    
    # Create markdown output directory
    md_output_dir = Path(config.output_dir) / config.language / "markdown_docs"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting documents to markdown in {md_output_dir}")
    
    # Group paragraphs by base docid
    # Format: {base_docid: {"title": str, "paragraphs": [(paragraph_id, text), ...]}}
    docs_by_id: dict[str, dict] = {}
    paragraph_count = 0
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            docid = doc["docid"]
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            # Parse docid: base_docid#paragraph_id
            if "#" in docid:
                base_docid, para_id = docid.rsplit("#", 1)
                try:
                    para_id = int(para_id)
                except ValueError:
                    para_id = 0
            else:
                base_docid = docid
                para_id = 0
            
            if base_docid not in docs_by_id:
                docs_by_id[base_docid] = {"title": title, "paragraphs": []}
            
            docs_by_id[base_docid]["paragraphs"].append((para_id, text))
            paragraph_count += 1
            
            if paragraph_count % 10000 == 0:
                context.log.info(f"Processed {paragraph_count} paragraphs...")
    
    context.log.info(f"Grouped {paragraph_count} paragraphs into {len(docs_by_id)} documents")
    
    # Write each document as a single markdown file
    doc_count = 0
    doc_index = {}  # Map base_docid to markdown file path
    
    for base_docid, doc_data in docs_by_id.items():
        title = doc_data["title"]
        paragraphs = doc_data["paragraphs"]
        
        # Sort paragraphs by paragraph_id
        paragraphs.sort(key=lambda x: x[0])
        
        # Concatenate paragraphs with double newline
        paragraph_texts = [p[1] for p in paragraphs]
        combined_text = "\n\n".join(paragraph_texts)
        
        # Create markdown content: title as H1, then all paragraphs
        md_content = f"# {title}\n\n{combined_text}\n"
        
        # Use base_docid as filename (sanitize for filesystem)
        safe_docid = base_docid.replace("/", "_").replace("\\", "_")
        md_file = md_output_dir / f"{safe_docid}.md"
        
        with open(md_file, "w", encoding="utf-8") as mf:
            mf.write(md_content)
        
        doc_index[base_docid] = str(md_file)
        doc_count += 1
        
        if doc_count % 10000 == 0:
            context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} documents to markdown")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "language": config.language,
        "document_count": doc_count,
        "paragraph_count": paragraph_count,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[miracl_markdown_docs, miracl_qrels],
    description="Build RAG evaluation dataset linking queries, documents, and relevance",
    group_name="miracl",
)
def miracl_rag_eval_dataset(context: AssetExecutionContext, config: MiraclConfig) -> dict[str, Any]:
    """Build a structured dataset for RAG recall evaluation.
    
    This asset creates a comprehensive evaluation dataset that links:
    - Queries (topics)
    - Document markdown files
    - Relevance judgments (qrels)
    
    Output format for each query:
    {
        "query_id": "...",
        "query": "...",
        "relevant_docs": [
            {
                "docid": "...",
                "markdown_path": "...",
                "title": "...",
                "text": "...",
                "relevance": 1
            },
            ...
        ],
        "non_relevant_docs": [...]
    }
    
    Returns:
        Dict containing evaluation dataset metadata.
    """
    base_path = Path(config.output_dir) / config.language
    
    # Load document index
    index_path = base_path / "markdown_docs" / "doc_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Document index not found: {index_path}. Run miracl_markdown_docs first.")
    
    with open(index_path, "r", encoding="utf-8") as f:
        doc_index = json.load(f)
    
    # Load topics
    topics_path = base_path / "qrels" / f"topics_{config.split}.jsonl"
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}. Run miracl_qrels first.")
    
    topics = {}
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            topic = json.loads(line)
            topics[topic["query_id"]] = topic["query"]
    
    # Load qrels and group by query_id
    qrels_path = base_path / "qrels" / f"qrels_{config.split}.jsonl"
    query_docs: dict[str, dict] = {}
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            qrel = json.loads(line)
            query_id = qrel["query_id"]
            
            if query_id not in query_docs:
                query_docs[query_id] = {
                    "relevant_docs": [],
                    "non_relevant_docs": [],
                }
            
            doc_entry = {
                "docid": qrel["docid"],
                "markdown_path": doc_index.get(qrel["docid"], ""),
                "title": qrel.get("title", ""),
                "text": qrel.get("text", ""),
                "relevance": qrel["relevance"],
            }
            
            if qrel["relevance"] > 0:
                query_docs[query_id]["relevant_docs"].append(doc_entry)
            else:
                query_docs[query_id]["non_relevant_docs"].append(doc_entry)
    
    # Build final evaluation dataset
    eval_dataset = []
    for query_id, docs_info in query_docs.items():
        eval_entry = {
            "query_id": query_id,
            "query": topics.get(query_id, ""),
            "relevant_docs": docs_info["relevant_docs"],
            "non_relevant_docs": docs_info["non_relevant_docs"],
        }
        eval_dataset.append(eval_entry)
    
    # Save evaluation dataset
    eval_output_dir = base_path / "rag_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_file = eval_output_dir / f"eval_dataset_{config.split}.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for entry in eval_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Also save a summary JSON for quick access
    summary_file = eval_output_dir / f"eval_summary_{config.split}.json"
    summary = {
        "language": config.language,
        "split": config.split,
        "total_queries": len(eval_dataset),
        "queries_with_relevant_docs": sum(1 for e in eval_dataset if e["relevant_docs"]),
        "total_relevant_judgments": sum(len(e["relevant_docs"]) for e in eval_dataset),
        "total_non_relevant_judgments": sum(len(e["non_relevant_docs"]) for e in eval_dataset),
        "eval_dataset_path": str(eval_file),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Built RAG evaluation dataset with {len(eval_dataset)} queries")
    context.log.info(f"Saved to {eval_file}")
    context.log.info(f"Summary saved to {summary_file}")
    
    return summary


# =============================================================================
# Enron Email Dataset Assets
# =============================================================================

class EnronConfig(Config):
    """Configuration for Enron email dataset assets."""

    output_dir: str = "output/enron"
    # CMU Enron email dataset URL
    dataset_url: str = "http://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    max_emails: int = 0  # 0 means no limit


def _parse_email_file(file_path: Path) -> dict[str, Any] | None:
    """Parse a single email file and extract its contents.
    
    Args:
        file_path: Path to the email file.
        
    Returns:
        Dict containing email metadata and content, or None if parsing fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f, policy=policy.default)
        
        # Extract email headers
        subject = msg.get("Subject", "") or ""
        from_addr = msg.get("From", "") or ""
        to_addr = msg.get("To", "") or ""
        cc_addr = msg.get("Cc", "") or ""
        date = msg.get("Date", "") or ""
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
            else:
                body = msg.get_payload() or ""
        
        return {
            "subject": subject,
            "from": from_addr,
            "to": to_addr,
            "cc": cc_addr,
            "date": date,
            "body": body.strip(),
            "file_path": str(file_path),
        }
    except Exception:
        return None


@asset(
    description="Download and extract Enron email dataset from CMU",
    group_name="enron",
)
def enron_emails(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Download and extract the Enron email dataset.
    
    The dataset contains approximately 500,000 emails from about 150 Enron employees.
    Source: Carnegie Mellon University CALO Project.
    
    Returns:
        Dict containing dataset metadata and paths.
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache_dir = output_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the tar.gz file
    tar_file = cache_dir / "enron_mail.tar.gz"
    
    if not tar_file.exists():
        context.log.info(f"Downloading Enron email dataset from {config.dataset_url}...")
        context.log.info("This may take a while (~400MB)...")
        urllib.request.urlretrieve(config.dataset_url, tar_file)
        context.log.info(f"Downloaded to {tar_file}")
    else:
        context.log.info(f"Using cached file: {tar_file}")
    
    # Extract the tar.gz file
    extract_dir = output_path / "maildir"
    
    if not extract_dir.exists():
        context.log.info("Extracting tar.gz file...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=output_path)
        context.log.info(f"Extracted to {output_path}")
    else:
        context.log.info(f"Using extracted data: {extract_dir}")
    
    # Parse all email files
    context.log.info("Parsing email files...")
    
    emails_list: list[dict] = []
    email_count = 0
    
    # Find the maildir (it might be nested)
    if (output_path / "maildir").exists():
        maildir = output_path / "maildir"
    elif (output_path / "enron_mail_20150507" / "maildir").exists():
        maildir = output_path / "enron_mail_20150507" / "maildir"
    else:
        # Search for maildir
        maildir = None
        for item in output_path.rglob("maildir"):
            if item.is_dir():
                maildir = item
                break
        if maildir is None:
            raise FileNotFoundError("Could not find maildir in extracted data")
    
    context.log.info(f"Found maildir at: {maildir}")
    
    # Iterate through all users and their emails
    for user_dir in sorted(maildir.iterdir()):
        if not user_dir.is_dir():
            continue
        
        user_name = user_dir.name
        
        # Iterate through all folders for this user
        for email_file in user_dir.rglob("*"):
            if not email_file.is_file():
                continue
            
            # Skip non-email files
            if email_file.suffix in [".DS_Store", ".gitignore"]:
                continue
            
            email_data = _parse_email_file(email_file)
            if email_data:
                email_data["user"] = user_name
                email_data["folder"] = str(email_file.parent.relative_to(user_dir))
                emails_list.append(email_data)
                email_count += 1
                
                if email_count % 10000 == 0:
                    context.log.info(f"Parsed {email_count} emails...")
                
                if config.max_emails > 0 and email_count >= config.max_emails:
                    context.log.info(f"Reached max_emails limit: {config.max_emails}")
                    break
        
        if config.max_emails > 0 and email_count >= config.max_emails:
            break
    
    context.log.info(f"Total emails parsed: {len(emails_list)}")
    
    # Save emails to JSONL
    emails_file = output_path / "emails.jsonl"
    with open(emails_file, "w", encoding="utf-8") as f:
        for email_data in emails_list:
            f.write(json.dumps(email_data, ensure_ascii=False) + "\n")
    
    context.log.info(f"Saved emails to {emails_file}")
    
    # Get unique users
    unique_users = set(e["user"] for e in emails_list)
    
    return {
        "email_count": len(emails_list),
        "unique_users": len(unique_users),
        "emails_path": str(emails_file),
        "maildir_path": str(maildir),
    }


@asset(
    deps=[enron_emails],
    description="Convert Enron emails to markdown files",
    group_name="enron",
)
def enron_markdown_docs(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Convert Enron emails to individual markdown files.
    
    Each email is saved as a markdown file with:
    - Subject as H1 header
    - Metadata (From, To, CC, Date) as a header section
    - Email body as content
    
    Returns:
        Dict containing markdown output metadata.
    """
    emails_path = Path(config.output_dir) / "emails.jsonl"
    
    if not emails_path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_path}. Run enron_emails first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_docs"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting emails to markdown in {md_output_dir}")
    
    doc_count = 0
    doc_index = {}
    
    with open(emails_path, "r", encoding="utf-8") as f:
        for line in f:
            email_data = json.loads(line)
            
            subject = email_data.get("subject", "") or "No Subject"
            from_addr = email_data.get("from", "")
            to_addr = email_data.get("to", "")
            cc_addr = email_data.get("cc", "")
            date = email_data.get("date", "")
            body = email_data.get("body", "")
            user = email_data.get("user", "")
            folder = email_data.get("folder", "")
            
            # Create markdown content
            md_lines = [
                f"# {subject}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| From | {from_addr} |",
                f"| To | {to_addr} |",
            ]
            
            if cc_addr:
                md_lines.append(f"| CC | {cc_addr} |")
            
            md_lines.extend([
                f"| Date | {date} |",
                f"| User | {user} |",
                f"| Folder | {folder} |",
                "",
                "---",
                "",
                body,
            ])
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use doc_count as filename to ensure uniqueness
            doc_id = f"{doc_count:06d}"
            md_file = md_output_dir / f"{doc_id}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[doc_id] = {
                "path": str(md_file),
                "subject": subject,
                "user": user,
                "folder": folder,
            }
            doc_count += 1
            
            if doc_count % 10000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} emails to markdown")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[enron_emails],
    description="Convert Enron emails to markdown files grouped by user",
    group_name="enron",
)
def enron_markdown_by_user(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Convert Enron emails to markdown files, organized by user.
    
    Creates a directory structure:
    - {user}/
      - {folder}/
        - {email_id}.md
    
    Returns:
        Dict containing markdown output metadata.
    """
    emails_path = Path(config.output_dir) / "emails.jsonl"
    
    if not emails_path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_path}. Run enron_emails first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_by_user"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting emails to markdown by user in {md_output_dir}")
    
    doc_count = 0
    user_email_counts: dict[str, int] = {}
    doc_index = {}
    
    with open(emails_path, "r", encoding="utf-8") as f:
        for line in f:
            email_data = json.loads(line)
            
            subject = email_data.get("subject", "") or "No Subject"
            from_addr = email_data.get("from", "")
            to_addr = email_data.get("to", "")
            cc_addr = email_data.get("cc", "")
            date = email_data.get("date", "")
            body = email_data.get("body", "")
            user = email_data.get("user", "unknown")
            folder = email_data.get("folder", "inbox")
            
            # Create user/folder directory
            user_dir = md_output_dir / user / folder
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Track email count per user for unique naming
            if user not in user_email_counts:
                user_email_counts[user] = 0
            user_email_counts[user] += 1
            email_num = user_email_counts[user]
            
            # Create markdown content
            md_lines = [
                f"# {subject}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| From | {from_addr} |",
                f"| To | {to_addr} |",
            ]
            
            if cc_addr:
                md_lines.append(f"| CC | {cc_addr} |")
            
            md_lines.extend([
                f"| Date | {date} |",
                "",
                "---",
                "",
                body,
            ])
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use email number as filename
            doc_id = f"{user}/{folder}/{email_num:05d}"
            md_file = user_dir / f"{email_num:05d}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[doc_id] = {
                "path": str(md_file),
                "subject": subject,
                "user": user,
                "folder": folder,
            }
            doc_count += 1
            
            if doc_count % 10000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} emails to markdown")
    context.log.info(f"Unique users: {len(user_email_counts)}")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "unique_users": len(user_email_counts),
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


# =============================================================================
# English Language Assets
# =============================================================================

class MiraclConfigEn(Config):
    """Configuration for MIRACL English dataset assets."""

    language: str = "en"
    split: str = "dev"  # train or dev
    output_dir: str = "output/miracl"


@asset(
    description="Download MIRACL English corpus from HuggingFace datasets",
    group_name="miracl_en",
)
def miracl_corpus_en(context: AssetExecutionContext, config: MiraclConfigEn) -> dict[str, Any]:
    """Download and cache the MIRACL English document corpus."""
    context.log.info(f"Downloading MIRACL corpus for language: {config.language}")
    
    cache_dir = Path(config.output_dir) / config.language / "cache" / "corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    repo_id = "miracl/miracl-corpus"
    
    context.log.info("Listing files in miracl/miracl-corpus...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    parquet_files = []
    for f in all_files:
        if f.endswith(".parquet") and config.language in f:
            parquet_files.append(f)
    
    context.log.info(f"Found {len(parquet_files)} parquet files for {config.language}")
    
    if not parquet_files:
        context.log.warning("No parquet files found, trying legacy load_dataset...")
        from datasets import load_dataset
        
        corpus = load_dataset(
            "miracl/miracl-corpus",
            config.language,
            split="train",
            trust_remote_code=True,
        )
        docs = list(corpus)
    else:
        docs = []
        for pf in parquet_files:
            context.log.info(f"Downloading {pf}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pf,
                repo_type="dataset",
            )
            df = pd.read_parquet(local_path)
            docs.extend(df.to_dict("records"))
        
        context.log.info(f"Loaded {len(docs)} documents from parquet files")
    
    output_path = Path(config.output_dir) / config.language / "corpus"
    output_path.mkdir(parents=True, exist_ok=True)
    
    corpus_file = output_path / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    context.log.info(f"Saved corpus to {corpus_file}")
    
    return {
        "language": config.language,
        "document_count": len(docs),
        "corpus_path": str(corpus_file),
    }


@asset(
    description="Download MIRACL English topics and qrels from HuggingFace datasets",
    group_name="miracl_en",
)
def miracl_qrels_en(context: AssetExecutionContext, config: MiraclConfigEn) -> dict[str, Any]:
    """Download English topics (queries) and relevance judgments (qrels)."""
    context.log.info(f"Downloading MIRACL qrels for language: {config.language}, split: {config.split}")
    
    repo_id = "miracl/miracl"
    
    context.log.info("Listing files in miracl/miracl...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    parquet_files = []
    for f in all_files:
        if f.endswith(".parquet") and config.language in f and config.split in f:
            parquet_files.append(f)
    
    context.log.info(f"Found {len(parquet_files)} parquet files for {config.language}/{config.split}")
    
    if not parquet_files:
        context.log.warning("No parquet files found, trying legacy load_dataset...")
        from datasets import load_dataset
        
        dataset = load_dataset(
            "miracl/miracl",
            config.language,
            split=config.split,
            trust_remote_code=True,
        )
        items = list(dataset)
    else:
        items = []
        for pf in parquet_files:
            context.log.info(f"Downloading {pf}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=pf,
                repo_type="dataset",
            )
            df = pd.read_parquet(local_path)
            items.extend(df.to_dict("records"))
        
        context.log.info(f"Loaded {len(items)} queries from parquet files")
    
    context.log.info(f"Loaded {len(items)} queries with relevance judgments")
    
    output_path = Path(config.output_dir) / config.language / "qrels"
    output_path.mkdir(parents=True, exist_ok=True)
    
    topics_file = output_path / f"topics_{config.split}.jsonl"
    qrels_file = output_path / f"qrels_{config.split}.jsonl"
    
    total_judgments = 0
    with open(topics_file, "w", encoding="utf-8") as tf, \
         open(qrels_file, "w", encoding="utf-8") as qf:
        for item in items:
            topic = {
                "query_id": item["query_id"],
                "query": item["query"],
            }
            tf.write(json.dumps(topic, ensure_ascii=False) + "\n")
            
            positive_passages = item.get("positive_passages") or []
            if isinstance(positive_passages, dict):
                docids = positive_passages.get("docid", [])
                titles = positive_passages.get("title", [])
                texts = positive_passages.get("text", [])
                for i in range(len(docids)):
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": docids[i] if i < len(docids) else "",
                        "relevance": 1,
                        "title": titles[i] if i < len(titles) else "",
                        "text": texts[i] if i < len(texts) else "",
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
            else:
                for passage in positive_passages:
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": passage["docid"],
                        "relevance": 1,
                        "title": passage.get("title", ""),
                        "text": passage.get("text", ""),
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
            
            negative_passages = item.get("negative_passages") or []
            if isinstance(negative_passages, dict):
                docids = negative_passages.get("docid", [])
                titles = negative_passages.get("title", [])
                texts = negative_passages.get("text", [])
                for i in range(len(docids)):
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": docids[i] if i < len(docids) else "",
                        "relevance": 0,
                        "title": titles[i] if i < len(titles) else "",
                        "text": texts[i] if i < len(texts) else "",
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
            else:
                for passage in negative_passages:
                    qrel = {
                        "query_id": item["query_id"],
                        "docid": passage["docid"],
                        "relevance": 0,
                        "title": passage.get("title", ""),
                        "text": passage.get("text", ""),
                    }
                    qf.write(json.dumps(qrel, ensure_ascii=False) + "\n")
                    total_judgments += 1
    
    context.log.info(f"Saved topics to {topics_file}")
    context.log.info(f"Saved qrels to {qrels_file}")
    
    return {
        "language": config.language,
        "split": config.split,
        "query_count": len(items),
        "judgment_count": total_judgments,
        "topics_path": str(topics_file),
        "qrels_path": str(qrels_file),
    }


@asset(
    deps=[miracl_qrels_en],
    description="Extract unique base docids from MIRACL English qrels",
    group_name="miracl_en",
)
def miracl_unique_docids_en(context: AssetExecutionContext, config: MiraclConfigEn) -> dict[str, Any]:
    """Extract all unique base docids from English qrels file."""
    qrels_path = Path(config.output_dir) / config.language / "qrels" / f"qrels_{config.split}.jsonl"
    
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}. Run miracl_qrels_en first.")
    
    context.log.info(f"Extracting unique docids from {qrels_path}")
    
    unique_docids: set[str] = set()
    total_entries = 0
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            qrel = json.loads(line)
            docid = qrel["docid"]
            
            if "#" in docid:
                base_docid = docid.split("#")[0]
            else:
                base_docid = docid
            
            unique_docids.add(base_docid)
            total_entries += 1
    
    context.log.info(f"Processed {total_entries} qrel entries")
    context.log.info(f"Found {len(unique_docids)} unique base docids")
    
    output_path = Path(config.output_dir) / config.language / "qrels"
    output_path.mkdir(parents=True, exist_ok=True)
    
    docids_file = output_path / f"unique_docids_{config.split}.json"
    sorted_docids = sorted(unique_docids, key=lambda x: int(x) if x.isdigit() else x)
    
    with open(docids_file, "w", encoding="utf-8") as f:
        json.dump({
            "unique_docid_count": len(unique_docids),
            "docids": sorted_docids,
        }, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Saved unique docids to {docids_file}")
    
    return {
        "language": config.language,
        "split": config.split,
        "total_qrel_entries": total_entries,
        "unique_docid_count": len(unique_docids),
        "docids_path": str(docids_file),
    }


@asset(
    deps=[miracl_unique_docids_en, miracl_corpus_en],
    description="Extract English corpus documents that match unique docids from qrels and save as markdown",
    group_name="miracl_en",
)
def miracl_filtered_corpus_en(context: AssetExecutionContext, config: MiraclConfigEn) -> dict[str, Any]:
    """Extract only English documents from corpus that have matching docids in qrels."""
    docids_path = Path(config.output_dir) / config.language / "qrels" / f"unique_docids_{config.split}.json"
    
    if not docids_path.exists():
        raise FileNotFoundError(f"Unique docids file not found: {docids_path}. Run miracl_unique_docids_en first.")
    
    with open(docids_path, "r", encoding="utf-8") as f:
        docids_data = json.load(f)
    
    unique_docids = set(docids_data["docids"])
    context.log.info(f"Loaded {len(unique_docids)} unique docids to filter")
    
    corpus_path = Path(config.output_dir) / config.language / "corpus" / "corpus.jsonl"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run miracl_corpus_en first.")
    
    context.log.info(f"Filtering corpus from {corpus_path}")
    
    docs_by_id: dict[str, dict] = {}
    total_paragraphs = 0
    filtered_paragraphs = 0
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            total_paragraphs += 1
            doc = json.loads(line)
            docid = doc["docid"]
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            if "#" in docid:
                base_docid, para_id = docid.rsplit("#", 1)
                try:
                    para_id = int(para_id)
                except ValueError:
                    para_id = 0
            else:
                base_docid = docid
                para_id = 0
            
            if base_docid in unique_docids:
                if base_docid not in docs_by_id:
                    docs_by_id[base_docid] = {"title": title, "paragraphs": []}
                
                docs_by_id[base_docid]["paragraphs"].append((para_id, text))
                filtered_paragraphs += 1
            
            if total_paragraphs % 100000 == 0:
                context.log.info(f"Processed {total_paragraphs} paragraphs, kept {filtered_paragraphs}...")
    
    context.log.info(f"Filtered {filtered_paragraphs} paragraphs from {total_paragraphs} total")
    context.log.info(f"Found {len(docs_by_id)} unique documents to write")
    
    md_output_dir = Path(config.output_dir) / config.language / "filtered_corpus" / config.split
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    doc_count = 0
    doc_index = {}
    
    for base_docid, doc_data in docs_by_id.items():
        title = doc_data["title"]
        paragraphs = doc_data["paragraphs"]
        
        paragraphs.sort(key=lambda x: x[0])
        
        paragraph_texts = [p[1] for p in paragraphs]
        combined_text = "\n\n".join(paragraph_texts)
        
        md_content = f"# {title}\n\n{combined_text}\n"
        
        # Use docid as filename (sanitize for filesystem)
        safe_docid = base_docid.replace("/", "_").replace("\\", "_")
        md_file = md_output_dir / f"{safe_docid}.md"
        
        with open(md_file, "w", encoding="utf-8") as mf:
            mf.write(md_content)
        
        doc_index[base_docid] = str(md_file)
        doc_count += 1
        
        if doc_count % 1000 == 0:
            context.log.info(f"Wrote {doc_count} markdown files...")
    
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Wrote {doc_count} markdown files to {md_output_dir}")
    context.log.info(f"Document index saved to {index_file}")
    
    matched_docids = set(docs_by_id.keys())
    missing_docids = unique_docids - matched_docids
    if missing_docids:
        context.log.warning(f"Warning: {len(missing_docids)} docids from qrels not found in corpus")
        missing_file = md_output_dir / "missing_docids.json"
        with open(missing_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(missing_docids)), f, ensure_ascii=False, indent=2)
        context.log.info(f"Missing docids saved to {missing_file}")
    
    return {
        "language": config.language,
        "split": config.split,
        "total_corpus_paragraphs": total_paragraphs,
        "filtered_paragraphs": filtered_paragraphs,
        "document_count": doc_count,
        "matched_docid_count": len(matched_docids),
        "missing_docid_count": len(missing_docids),
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[miracl_corpus_en],
    description="Convert MIRACL English corpus documents to markdown files",
    group_name="miracl_en",
)
def miracl_markdown_docs_en(context: AssetExecutionContext, config: MiraclConfigEn) -> dict[str, Any]:
    """Convert MIRACL English documents to individual markdown files."""
    corpus_path = Path(config.output_dir) / config.language / "corpus" / "corpus.jsonl"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run miracl_corpus_en first.")
    
    md_output_dir = Path(config.output_dir) / config.language / "markdown_docs"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting documents to markdown in {md_output_dir}")
    
    docs_by_id: dict[str, dict] = {}
    paragraph_count = 0
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            docid = doc["docid"]
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            if "#" in docid:
                base_docid, para_id = docid.rsplit("#", 1)
                try:
                    para_id = int(para_id)
                except ValueError:
                    para_id = 0
            else:
                base_docid = docid
                para_id = 0
            
            if base_docid not in docs_by_id:
                docs_by_id[base_docid] = {"title": title, "paragraphs": []}
            
            docs_by_id[base_docid]["paragraphs"].append((para_id, text))
            paragraph_count += 1
            
            if paragraph_count % 10000 == 0:
                context.log.info(f"Processed {paragraph_count} paragraphs...")
    
    context.log.info(f"Grouped {paragraph_count} paragraphs into {len(docs_by_id)} documents")
    
    doc_count = 0
    doc_index = {}
    
    for base_docid, doc_data in docs_by_id.items():
        title = doc_data["title"]
        paragraphs = doc_data["paragraphs"]
        
        paragraphs.sort(key=lambda x: x[0])
        
        paragraph_texts = [p[1] for p in paragraphs]
        combined_text = "\n\n".join(paragraph_texts)
        
        md_content = f"# {title}\n\n{combined_text}\n"
        
        safe_docid = base_docid.replace("/", "_").replace("\\", "_")
        md_file = md_output_dir / f"{safe_docid}.md"
        
        with open(md_file, "w", encoding="utf-8") as mf:
            mf.write(md_content)
        
        doc_index[base_docid] = str(md_file)
        doc_count += 1
        
        if doc_count % 10000 == 0:
            context.log.info(f"Wrote {doc_count} markdown files...")
    
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} documents to markdown")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "language": config.language,
        "document_count": doc_count,
        "paragraph_count": paragraph_count,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[miracl_markdown_docs_en, miracl_qrels_en],
    description="Build English RAG evaluation dataset linking queries, documents, and relevance",
    group_name="miracl_en",
)
def miracl_rag_eval_dataset_en(context: AssetExecutionContext, config: MiraclConfigEn) -> dict[str, Any]:
    """Build a structured English dataset for RAG recall evaluation."""
    base_path = Path(config.output_dir) / config.language
    
    index_path = base_path / "markdown_docs" / "doc_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Document index not found: {index_path}. Run miracl_markdown_docs_en first.")
    
    with open(index_path, "r", encoding="utf-8") as f:
        doc_index = json.load(f)
    
    topics_path = base_path / "qrels" / f"topics_{config.split}.jsonl"
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}. Run miracl_qrels_en first.")
    
    topics = {}
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            topic = json.loads(line)
            topics[topic["query_id"]] = topic["query"]
    
    qrels_path = base_path / "qrels" / f"qrels_{config.split}.jsonl"
    query_docs: dict[str, dict] = {}
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            qrel = json.loads(line)
            query_id = qrel["query_id"]
            
            if query_id not in query_docs:
                query_docs[query_id] = {
                    "relevant_docs": [],
                    "non_relevant_docs": [],
                }
            
            doc_entry = {
                "docid": qrel["docid"],
                "markdown_path": doc_index.get(qrel["docid"], ""),
                "title": qrel.get("title", ""),
                "text": qrel.get("text", ""),
                "relevance": qrel["relevance"],
            }
            
            if qrel["relevance"] > 0:
                query_docs[query_id]["relevant_docs"].append(doc_entry)
            else:
                query_docs[query_id]["non_relevant_docs"].append(doc_entry)
    
    eval_dataset = []
    for query_id, docs_info in query_docs.items():
        eval_entry = {
            "query_id": query_id,
            "query": topics.get(query_id, ""),
            "relevant_docs": docs_info["relevant_docs"],
            "non_relevant_docs": docs_info["non_relevant_docs"],
        }
        eval_dataset.append(eval_entry)
    
    eval_output_dir = base_path / "rag_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_file = eval_output_dir / f"eval_dataset_{config.split}.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for entry in eval_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    summary_file = eval_output_dir / f"eval_summary_{config.split}.json"
    summary = {
        "language": config.language,
        "split": config.split,
        "total_queries": len(eval_dataset),
        "queries_with_relevant_docs": sum(1 for e in eval_dataset if e["relevant_docs"]),
        "total_relevant_judgments": sum(len(e["relevant_docs"]) for e in eval_dataset),
        "total_non_relevant_judgments": sum(len(e["non_relevant_docs"]) for e in eval_dataset),
        "eval_dataset_path": str(eval_file),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Built RAG evaluation dataset with {len(eval_dataset)} queries")
    context.log.info(f"Saved to {eval_file}")
    context.log.info(f"Summary saved to {summary_file}")
    
    return summary


# =============================================================================
# Enron Email Dataset Assets
# =============================================================================

class EnronConfig(Config):
    """Configuration for Enron email dataset assets."""

    output_dir: str = "output/enron"
    # CMU Enron email dataset URL
    dataset_url: str = "http://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    max_emails: int = 0  # 0 means no limit


def _parse_email_file(file_path: Path) -> dict[str, Any] | None:
    """Parse a single email file and extract its contents.
    
    Args:
        file_path: Path to the email file.
        
    Returns:
        Dict containing email metadata and content, or None if parsing fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f, policy=policy.default)
        
        # Extract email headers
        subject = msg.get("Subject", "") or ""
        from_addr = msg.get("From", "") or ""
        to_addr = msg.get("To", "") or ""
        cc_addr = msg.get("Cc", "") or ""
        date = msg.get("Date", "") or ""
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
            else:
                body = msg.get_payload() or ""
        
        return {
            "subject": subject,
            "from": from_addr,
            "to": to_addr,
            "cc": cc_addr,
            "date": date,
            "body": body.strip(),
            "file_path": str(file_path),
        }
    except Exception:
        return None


@asset(
    description="Download and extract Enron email dataset from CMU",
    group_name="enron",
)
def enron_emails(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Download and extract the Enron email dataset.
    
    The dataset contains approximately 500,000 emails from about 150 Enron employees.
    Source: Carnegie Mellon University CALO Project.
    
    Returns:
        Dict containing dataset metadata and paths.
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache_dir = output_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the tar.gz file
    tar_file = cache_dir / "enron_mail.tar.gz"
    
    if not tar_file.exists():
        context.log.info(f"Downloading Enron email dataset from {config.dataset_url}...")
        context.log.info("This may take a while (~400MB)...")
        urllib.request.urlretrieve(config.dataset_url, tar_file)
        context.log.info(f"Downloaded to {tar_file}")
    else:
        context.log.info(f"Using cached file: {tar_file}")
    
    # Extract the tar.gz file
    extract_dir = output_path / "maildir"
    
    if not extract_dir.exists():
        context.log.info("Extracting tar.gz file...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=output_path)
        context.log.info(f"Extracted to {output_path}")
    else:
        context.log.info(f"Using extracted data: {extract_dir}")
    
    # Parse all email files
    context.log.info("Parsing email files...")
    
    emails_list: list[dict] = []
    email_count = 0
    
    # Find the maildir (it might be nested)
    if (output_path / "maildir").exists():
        maildir = output_path / "maildir"
    elif (output_path / "enron_mail_20150507" / "maildir").exists():
        maildir = output_path / "enron_mail_20150507" / "maildir"
    else:
        # Search for maildir
        maildir = None
        for item in output_path.rglob("maildir"):
            if item.is_dir():
                maildir = item
                break
        if maildir is None:
            raise FileNotFoundError("Could not find maildir in extracted data")
    
    context.log.info(f"Found maildir at: {maildir}")
    
    # Iterate through all users and their emails
    for user_dir in sorted(maildir.iterdir()):
        if not user_dir.is_dir():
            continue
        
        user_name = user_dir.name
        
        # Iterate through all folders for this user
        for email_file in user_dir.rglob("*"):
            if not email_file.is_file():
                continue
            
            # Skip non-email files
            if email_file.suffix in [".DS_Store", ".gitignore"]:
                continue
            
            email_data = _parse_email_file(email_file)
            if email_data:
                email_data["user"] = user_name
                email_data["folder"] = str(email_file.parent.relative_to(user_dir))
                emails_list.append(email_data)
                email_count += 1
                
                if email_count % 10000 == 0:
                    context.log.info(f"Parsed {email_count} emails...")
                
                if config.max_emails > 0 and email_count >= config.max_emails:
                    context.log.info(f"Reached max_emails limit: {config.max_emails}")
                    break
        
        if config.max_emails > 0 and email_count >= config.max_emails:
            break
    
    context.log.info(f"Total emails parsed: {len(emails_list)}")
    
    # Save emails to JSONL
    emails_file = output_path / "emails.jsonl"
    with open(emails_file, "w", encoding="utf-8") as f:
        for email_data in emails_list:
            f.write(json.dumps(email_data, ensure_ascii=False) + "\n")
    
    context.log.info(f"Saved emails to {emails_file}")
    
    # Get unique users
    unique_users = set(e["user"] for e in emails_list)
    
    return {
        "email_count": len(emails_list),
        "unique_users": len(unique_users),
        "emails_path": str(emails_file),
        "maildir_path": str(maildir),
    }


@asset(
    deps=[enron_emails],
    description="Convert Enron emails to markdown files",
    group_name="enron",
)
def enron_markdown_docs(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Convert Enron emails to individual markdown files.
    
    Each email is saved as a markdown file with:
    - Subject as H1 header
    - Metadata (From, To, CC, Date) as a header section
    - Email body as content
    
    Returns:
        Dict containing markdown output metadata.
    """
    emails_path = Path(config.output_dir) / "emails.jsonl"
    
    if not emails_path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_path}. Run enron_emails first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_docs"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting emails to markdown in {md_output_dir}")
    
    doc_count = 0
    doc_index = {}
    
    with open(emails_path, "r", encoding="utf-8") as f:
        for line in f:
            email_data = json.loads(line)
            
            subject = email_data.get("subject", "") or "No Subject"
            from_addr = email_data.get("from", "")
            to_addr = email_data.get("to", "")
            cc_addr = email_data.get("cc", "")
            date = email_data.get("date", "")
            body = email_data.get("body", "")
            user = email_data.get("user", "")
            folder = email_data.get("folder", "")
            
            # Create markdown content
            md_lines = [
                f"# {subject}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| From | {from_addr} |",
                f"| To | {to_addr} |",
            ]
            
            if cc_addr:
                md_lines.append(f"| CC | {cc_addr} |")
            
            md_lines.extend([
                f"| Date | {date} |",
                f"| User | {user} |",
                f"| Folder | {folder} |",
                "",
                "---",
                "",
                body,
            ])
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use doc_count as filename to ensure uniqueness
            doc_id = f"{doc_count:06d}"
            md_file = md_output_dir / f"{doc_id}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[doc_id] = {
                "path": str(md_file),
                "subject": subject,
                "user": user,
                "folder": folder,
            }
            doc_count += 1
            
            if doc_count % 10000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} emails to markdown")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[enron_emails],
    description="Convert Enron emails to markdown files grouped by user",
    group_name="enron",
)
def enron_markdown_by_user(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Convert Enron emails to markdown files, organized by user.
    
    Creates a directory structure:
    - {user}/
      - {folder}/
        - {email_id}.md
    
    Returns:
        Dict containing markdown output metadata.
    """
    emails_path = Path(config.output_dir) / "emails.jsonl"
    
    if not emails_path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_path}. Run enron_emails first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_by_user"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting emails to markdown by user in {md_output_dir}")
    
    doc_count = 0
    user_email_counts: dict[str, int] = {}
    doc_index = {}
    
    with open(emails_path, "r", encoding="utf-8") as f:
        for line in f:
            email_data = json.loads(line)
            
            subject = email_data.get("subject", "") or "No Subject"
            from_addr = email_data.get("from", "")
            to_addr = email_data.get("to", "")
            cc_addr = email_data.get("cc", "")
            date = email_data.get("date", "")
            body = email_data.get("body", "")
            user = email_data.get("user", "unknown")
            folder = email_data.get("folder", "inbox")
            
            # Create user/folder directory
            user_dir = md_output_dir / user / folder
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Track email count per user for unique naming
            if user not in user_email_counts:
                user_email_counts[user] = 0
            user_email_counts[user] += 1
            email_num = user_email_counts[user]
            
            # Create markdown content
            md_lines = [
                f"# {subject}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| From | {from_addr} |",
                f"| To | {to_addr} |",
            ]
            
            if cc_addr:
                md_lines.append(f"| CC | {cc_addr} |")
            
            md_lines.extend([
                f"| Date | {date} |",
                "",
                "---",
                "",
                body,
            ])
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use email number as filename
            doc_id = f"{user}/{folder}/{email_num:05d}"
            md_file = user_dir / f"{email_num:05d}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[doc_id] = {
                "path": str(md_file),
                "subject": subject,
                "user": user,
                "folder": folder,
            }
            doc_count += 1
            
            if doc_count % 10000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} emails to markdown")
    context.log.info(f"Unique users: {len(user_email_counts)}")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "unique_users": len(user_email_counts),
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }
