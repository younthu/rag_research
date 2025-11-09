#!/usr/bin/env python3
"""
ingest_to_es.py

python3 scripts/ingest_to_es.py -i downloads/toutiao_cat_data.txt --index toutiao_news --es-host http://localhost:9200 --batch 500

python3 scripts/ingest_to_es.py -i downloads/toutiao_cat_data.txt --index dummy --csv downloads/toutiao_cat_data.csv --limit 1000

Usage:
  python3 scripts/ingest_to_es.py \
    --input downloads/toutiao_cat_data.txt \
    --index toutiao_news \
    --es-host http://localhost:9200 \
    [--sep "_!_"] [--csv downloads/toutiao_cat_data.csv] [--batch 500]

This script:
  1. Parses the input dataset (default separator "_!_") into records with fields:
       id, category_id, label, title, keywords
  2. Optionally writes a CSV file
  3. Bulk indexes the records into the given Elasticsearch index

Notes:
  - The script will create the index with a simple mapping if it doesn't exist.
  - For local dev the script assumes no authentication. To use auth, set ES environment
    variables or modify the Elasticsearch client initialization below.
"""
from __future__ import annotations

import argparse
import csv
import sys
from typing import Dict, Iterable

try:
    from elasticsearch import Elasticsearch, helpers
except Exception as e:
    print("elasticsearch package is required. Install with: pip install elasticsearch", file=sys.stderr)
    raise


SEP_DEFAULT = "_!_"


def parse_line(line: str, sep: str = SEP_DEFAULT):
    parts = line.rstrip('\n').split(sep)
    parts = [p.strip() for p in parts]
    # Expected up to 5 fields. Pad or join extras into last.
    if len(parts) < 4:
        # skip malformed lines
        return None
    if len(parts) < 5:
        parts += [''] * (5 - len(parts))
    if len(parts) > 5:
        parts = parts[:4] + [sep.join(parts[4:])]
    id_, cat_id, label, title, keywords = parts[:5]
    # Normalize category id
    try:
        cat_id_int = int(cat_id)
    except Exception:
        cat_id_int = None
    return {
        'id': id_,
        'category_id': cat_id_int,
        'label': label,
        'title': title,
        'keywords': keywords,
    }


def read_records(path: str, sep: str = SEP_DEFAULT, limit: int = 0) -> Iterable[Dict]:
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
            if not line.strip():
                continue
            rec = parse_line(line, sep=sep)
            if rec is None:
                continue
            yield rec
            count += 1


def ensure_index(es: Elasticsearch, index: str):
    if es.indices.exists(index=index):
        print(f"Index '{index}' already exists")
        return
    mapping = {
        'mappings': {
            'properties': {
                'id': {'type': 'keyword'},
                'category_id': {'type': 'integer'},
                'label': {'type': 'keyword'},
                'title': {'type': 'text', 'analyzer': 'standard'},
                'keywords': {'type': 'keyword'},
            }
        }
    }
    es.indices.create(index=index, body=mapping)
    print(f"Created index '{index}' with basic mapping")


def docs_to_actions(docs: Iterable[Dict], index: str):
    for d in docs:
        # Use original id as _id
        yield {
            '_index': index,
            '_id': d.get('id'),
            '_source': d,
        }


def write_csv(path: str, docs: Iterable[Dict]):
    with open(path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id', 'category_id', 'label', 'title', 'keywords'])
        for d in docs:
            writer.writerow([d.get('id', ''), d.get('category_id', ''), d.get('label', ''), d.get('title', ''), d.get('keywords', '')])


def bulk_index(es: Elasticsearch, docs_iter: Iterable[Dict], index: str, batch_size: int = 500):
    actions = docs_to_actions(docs_iter, index)
    success, failed = 0, 0
    for ok, item in helpers.streaming_bulk(es, actions, chunk_size=batch_size):
        if ok:
            success += 1
        else:
            failed += 1
    return success, failed


def main():
    parser = argparse.ArgumentParser(description='Parse and ingest Toutiao dataset into Elasticsearch')
    parser.add_argument('--input', '-i', required=True, help='Path to input txt file')
    parser.add_argument('--index', required=True, help='Elasticsearch index name to create/use')
    parser.add_argument('--es-host', default='http://localhost:9200', help='Elasticsearch host URL')
    parser.add_argument('--sep', default=SEP_DEFAULT, help='Field separator token (default "_!_")')
    parser.add_argument('--csv', help='Optional: write parsed CSV to this path')
    parser.add_argument('--batch', type=int, default=500, help='Bulk batch size')
    parser.add_argument('--limit', type=int, default=0, help='Optional: limit number of lines to ingest (0 = all)')
    args = parser.parse_args()

    es = Elasticsearch([args.es_host])
    # quick ping
    try:
        if not es.ping():
            print(f"Cannot connect to Elasticsearch at {args.es_host}", file=sys.stderr)
            sys.exit(3)
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}", file=sys.stderr)
        sys.exit(3)

    ensure_index(es, args.index)

    # Read records generator
    records = list(read_records(args.input, sep=args.sep, limit=args.limit))
    print(f"Parsed {len(records)} records from {args.input}")

    if args.csv:
        write_csv(args.csv, records)
        print(f"Wrote CSV to {args.csv}")

    if len(records) == 0:
        print("No records to index. Exiting.")
        return

    # Bulk index
    print(f"Indexing into Elasticsearch index '{args.index}' (batch={args.batch})")
    success, failed = 0, 0
    # helpers.bulk accepts generators but we already have list; create iterator
    actions = docs_to_actions(iter(records), args.index)
    try:
        for ok, item in helpers.streaming_bulk(es, actions, chunk_size=args.batch):
            if ok:
                success += 1
            else:
                failed += 1
    except Exception as e:
        print(f"Bulk indexing error: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"Bulk indexing finished. success={success}, failed={failed}")


if __name__ == '__main__':
    main()
