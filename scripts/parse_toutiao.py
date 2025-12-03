#!/usr/bin/env python3
"""
parse_toutiao.py


Simple parser for the `toutiao_cat_data.txt` dataset.

Each line is separated by the token `_!_` and fields are:
  1) id
  2) category_id (numeric)
  3) label (string, e.g. news_culture)
  4) title
  5) keywords (comma separated, may be empty)

The script writes a CSV (UTF-8) with headers:
  id,category_id,label,title,keywords

Usage:
  python3 scripts/parse_toutiao.py --input downloads/toutiao_cat_data.txt --output downloads/toutiao_cat_data.csv

The script is memory-efficient and streams the input.
"""
import argparse
import csv
import sys


SEP = "_!_"


def parse_line(line):
    parts = line.rstrip('\n').split(SEP)
    # Trim whitespace
    parts = [p.strip() for p in parts]
    # Expected up to 5 fields. If fewer, pad with empty strings.
    if len(parts) < 5:
        parts += [''] * (5 - len(parts))
    # If there are more than 5 (rare), join extras into the last field
    if len(parts) > 5:
        parts = parts[:4] + [SEP.join(parts[4:])]
    return parts[:5]


def main():
    parser = argparse.ArgumentParser(description='Parse toutiao dataset into CSV')
    parser.add_argument('--input', '-i', required=True, help='Path to toutiao_cat_data.txt')
    parser.add_argument('--output', '-o', required=True, help='CSV output path')
    parser.add_argument('--limit', type=int, default=0, help='Optional: max number of lines to parse (0 = all)')
    args = parser.parse_args()

    try:
        fin = open(args.input, 'r', encoding='utf-8')
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    fout = open(args.output, 'w', encoding='utf-8', newline='')
    writer = csv.writer(fout)
    writer.writerow(['id', 'category_id', 'label', 'title', 'keywords'])

    count = 0
    for line in fin:
        if args.limit and count >= args.limit:
            break
        if not line.strip():
            continue
        id_, cat_id, label, title, keywords = parse_line(line)
        writer.writerow([id_, cat_id, label, title, keywords])
        count += 1
        if count % 50000 == 0:
            print(f"Parsed {count} lines...", file=sys.stderr)

    fin.close()
    fout.close()
    print(f"Done. Parsed {count} lines to {args.output}")


if __name__ == '__main__':
    main()
