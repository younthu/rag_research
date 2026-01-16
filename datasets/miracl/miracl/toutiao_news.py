"""Toutiao News Classification Dataset Assets.

This module provides Dagster assets for:
1. Downloading Toutiao news classification dataset from Huggingface
2. Parsing and processing the news data
3. Converting to markdown files
4. Splitting into train/dev/test sets
"""

import json
import random
from pathlib import Path
from typing import Any

from dagster import asset, AssetExecutionContext, Config
from huggingface_hub import hf_hub_download


# Category mapping for Toutiao dataset
TOUTIAO_CATEGORIES = {
    "100": "民生",      # news_story
    "101": "文化",      # news_culture
    "102": "娱乐",      # news_entertainment
    "103": "体育",      # news_sports
    "104": "财经",      # news_finance
    "105": "房产",      # news_house (removed in some versions)
    "106": "汽车",      # news_car
    "107": "教育",      # news_edu
    "108": "科技",      # news_tech
    "109": "军事",      # news_military
    "110": "旅游",      # news_travel
    "111": "国际",      # news_world
    "112": "证券",      # stock (removed in some versions)
    "113": "农业",      # news_agriculture
    "114": "电竞",      # news_game
}


class ToutiaoConfig(Config):
    """Configuration for Toutiao news classification dataset assets."""

    output_dir: str = "output/toutiao"
    # Huggingface dataset repo for toutiao_cat_data.txt
    hf_repo_id: str = "fourteenBDr/toutiao"
    hf_filename: str = "toutiao_cat_data.txt"
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1


@asset(
    description="Download Toutiao news classification dataset from Huggingface",
    group_name="toutiao",
)
def toutiao_download(context: AssetExecutionContext, config: ToutiaoConfig) -> dict[str, Any]:
    """Download the Toutiao news classification dataset from Huggingface.
    
    The dataset contains ~380,000 Chinese news headlines with 15 category labels.
    Source: https://huggingface.co/datasets/fourteenBDr/toutiao
    
    Returns:
        Dict containing download metadata and file path.
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache_dir = output_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = cache_dir / config.hf_filename
    
    if not data_file.exists():
        context.log.info(f"Downloading {config.hf_filename} from Huggingface...")
        context.log.info(f"Repository: {config.hf_repo_id}")
        
        # Download from Huggingface Hub
        downloaded_path = hf_hub_download(
            repo_id=config.hf_repo_id,
            filename=config.hf_filename,
            repo_type="dataset",
            local_dir=cache_dir,
        )
        
        context.log.info(f"Downloaded to {downloaded_path}")
        data_file = Path(downloaded_path)
    else:
        context.log.info(f"Using cached file: {data_file}")
    
    # Count lines
    line_count = 0
    with open(data_file, "r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1
    
    context.log.info(f"Dataset contains {line_count} records")
    
    return {
        "data_file": str(data_file),
        "line_count": line_count,
        "repo_id": config.hf_repo_id,
    }


def _parse_toutiao_line(line: str) -> dict[str, Any] | None:
    """Parse a single line from toutiao_cat_data.txt.
    
    Format: news_id_!_category_code_!_category_name_!_title_!_keywords
    
    Args:
        line: A single line from the data file.
        
    Returns:
        Dict containing parsed fields, or None if parsing fails.
    """
    parts = line.strip().split("_!_")
    if len(parts) != 5:
        return None
    
    news_id, cat_code, cat_name, title, keywords = parts
    
    # Get Chinese category name
    cat_name_zh = TOUTIAO_CATEGORIES.get(cat_code, cat_name)
    
    return {
        "news_id": news_id,
        "category_code": cat_code,
        "category_name": cat_name,
        "category_name_zh": cat_name_zh,
        "title": title,
        "keywords": keywords,
    }


@asset(
    deps=[toutiao_download],
    description="Parse Toutiao news classification dataset",
    group_name="toutiao",
)
def toutiao_news(context: AssetExecutionContext, config: ToutiaoConfig) -> dict[str, Any]:
    """Parse the Toutiao news classification dataset.
    
    The dataset contains Chinese news headlines with category labels.
    Format: news_id_!_category_code_!_category_name_!_title_!_keywords
    
    Returns:
        Dict containing dataset metadata and paths.
    """
    # Use downloaded file from cache
    data_path = Path(config.output_dir) / "cache" / config.hf_filename
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Run toutiao_download first."
        )
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Parsing Toutiao dataset from {data_path}")
    
    news_list: list[dict] = []
    category_counts: dict[str, int] = {}
    parse_errors = 0
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            parsed = _parse_toutiao_line(line)
            if parsed:
                news_list.append(parsed)
                cat = parsed["category_name_zh"]
                category_counts[cat] = category_counts.get(cat, 0) + 1
            else:
                parse_errors += 1
                if parse_errors <= 5:
                    context.log.warning(f"Parse error at line {line_num}: {line[:100]}...")
            
            if len(news_list) % 50000 == 0:
                context.log.info(f"Parsed {len(news_list)} news items...")
    
    context.log.info(f"Total news parsed: {len(news_list)}")
    context.log.info(f"Parse errors: {parse_errors}")
    context.log.info(f"Categories found: {len(category_counts)}")
    
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        context.log.info(f"  {cat}: {count}")
    
    # Save all news to JSONL
    news_file = output_path / "news.jsonl"
    with open(news_file, "w", encoding="utf-8") as f:
        for news in news_list:
            f.write(json.dumps(news, ensure_ascii=False) + "\n")
    
    context.log.info(f"Saved news to {news_file}")
    
    # Save category mapping
    categories_file = output_path / "categories.json"
    with open(categories_file, "w", encoding="utf-8") as f:
        json.dump({
            "category_counts": category_counts,
            "category_mapping": TOUTIAO_CATEGORIES,
        }, f, ensure_ascii=False, indent=2)
    
    return {
        "news_count": len(news_list),
        "category_count": len(category_counts),
        "parse_errors": parse_errors,
        "news_path": str(news_file),
        "categories_path": str(categories_file),
    }


@asset(
    deps=[toutiao_news],
    description="Convert Toutiao news to markdown files",
    group_name="toutiao",
)
def toutiao_markdown_docs(context: AssetExecutionContext, config: ToutiaoConfig) -> dict[str, Any]:
    """Convert Toutiao news to individual markdown files.
    
    Each news item is saved as a markdown file with:
    - Title as H1 header
    - Category and keywords as metadata
    
    Returns:
        Dict containing markdown output metadata.
    """
    news_path = Path(config.output_dir) / "news.jsonl"
    
    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}. Run toutiao_news first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_docs"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting news to markdown in {md_output_dir}")
    
    doc_count = 0
    doc_index = {}
    
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            news = json.loads(line)
            
            news_id = news["news_id"]
            title = news["title"]
            category = news["category_name_zh"]
            keywords = news["keywords"]
            
            # Create markdown content
            md_lines = [
                f"# {title}",
                "",
                f"**分类**: {category}",
                "",
                f"**关键词**: {keywords}",
            ]
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use news_id as filename
            md_file = md_output_dir / f"{news_id}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[news_id] = {
                "path": str(md_file),
                "title": title,
                "category": category,
            }
            doc_count += 1
            
            if doc_count % 50000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} news to markdown")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[toutiao_news],
    description="Convert Toutiao news to markdown files grouped by category",
    group_name="toutiao",
)
def toutiao_markdown_by_category(context: AssetExecutionContext, config: ToutiaoConfig) -> dict[str, Any]:
    """Convert Toutiao news to markdown files, organized by category.
    
    Creates a directory structure:
    - {category}/
      - {news_id}.md
    
    Returns:
        Dict containing markdown output metadata.
    """
    news_path = Path(config.output_dir) / "news.jsonl"
    
    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}. Run toutiao_news first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_by_category"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting news to markdown by category in {md_output_dir}")
    
    doc_count = 0
    category_counts: dict[str, int] = {}
    doc_index = {}
    
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            news = json.loads(line)
            
            news_id = news["news_id"]
            title = news["title"]
            category = news["category_name_zh"]
            keywords = news["keywords"]
            
            # Create category directory
            cat_dir = md_output_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            # Track count per category
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Create markdown content
            md_lines = [
                f"# {title}",
                "",
                f"**关键词**: {keywords}",
            ]
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use news_id as filename
            md_file = cat_dir / f"{news_id}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_id = f"{category}/{news_id}"
            doc_index[doc_id] = {
                "path": str(md_file),
                "title": title,
                "category": category,
            }
            doc_count += 1
            
            if doc_count % 50000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} news to markdown")
    context.log.info(f"Categories: {len(category_counts)}")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "category_count": len(category_counts),
        "category_counts": category_counts,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[toutiao_news],
    description="Split Toutiao dataset into train/dev/test sets",
    group_name="toutiao",
)
def toutiao_splits(context: AssetExecutionContext, config: ToutiaoConfig) -> dict[str, Any]:
    """Split Toutiao dataset into train/dev/test sets.
    
    Uses stratified sampling to maintain category distribution across splits.
    
    Returns:
        Dict containing split file paths and statistics.
    """
    news_path = Path(config.output_dir) / "news.jsonl"
    
    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}. Run toutiao_news first.")
    
    output_path = Path(config.output_dir) / "splits"
    output_path.mkdir(parents=True, exist_ok=True)
    
    context.log.info("Loading news data for splitting...")
    
    # Group news by category for stratified split
    news_by_category: dict[str, list[dict]] = {}
    
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            news = json.loads(line)
            cat = news["category_name_zh"]
            if cat not in news_by_category:
                news_by_category[cat] = []
            news_by_category[cat].append(news)
    
    context.log.info(f"Loaded {sum(len(v) for v in news_by_category.values())} news items")
    
    # Split each category
    train_data: list[dict] = []
    dev_data: list[dict] = []
    test_data: list[dict] = []
    
    for cat, news_list in news_by_category.items():
        random.seed(42)  # For reproducibility
        random.shuffle(news_list)
        
        n = len(news_list)
        train_end = int(n * config.train_ratio)
        dev_end = train_end + int(n * config.dev_ratio)
        
        train_data.extend(news_list[:train_end])
        dev_data.extend(news_list[train_end:dev_end])
        test_data.extend(news_list[dev_end:])
    
    # Shuffle final datasets
    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    context.log.info(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    # Save splits
    splits_info = {}
    for split_name, split_data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        split_file = output_path / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for news in split_data:
                f.write(json.dumps(news, ensure_ascii=False) + "\n")
        
        splits_info[split_name] = {
            "count": len(split_data),
            "path": str(split_file),
        }
        context.log.info(f"Saved {split_name} to {split_file}")
    
    # Save split summary
    summary_file = output_path / "splits_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(splits_info, f, ensure_ascii=False, indent=2)
    
    return {
        "train_count": len(train_data),
        "dev_count": len(dev_data),
        "test_count": len(test_data),
        "splits_dir": str(output_path),
        "splits_info": splits_info,
    }
