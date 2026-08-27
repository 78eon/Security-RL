#!/usr/bin/env python3
"""Merge saved Nmap and Greenbone XML into one live knowledge-graph snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.enterprise.live import LiveKnowledgeGraph, parse_greenbone_xml, parse_nmap_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmap-xml", type=Path, action="append", default=[])
    parser.add_argument("--greenbone-xml", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.nmap_xml and not args.greenbone_xml:
        parser.error("at least one evidence file is required")

    graph = LiveKnowledgeGraph()
    for path in args.nmap_xml:
        graph.ingest_nmap(parse_nmap_xml(path.read_bytes()))
    for path in args.greenbone_xml:
        graph.ingest_greenbone(parse_greenbone_xml(path.read_bytes()))
    rendered = json.dumps(graph.to_dict(), indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
