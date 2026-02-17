#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick Start Script for HRSC Framework

This script provides a simple way to test the HRSC framework with sample queries.
"""

import logging
from hrsc import initialize_hrsc_from_config
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Quick start demo"""
    print("\n" + "="*80)
    print("HRSC FRAMEWORK - QUICK START")
    print("="*80)
    
    # Initialize HRSC
    print("\n[1] Initializing HRSC Framework...")
    hrsc = initialize_hrsc_from_config()
    print("[OK] HRSC initialized successfully")
    
    # Index sample document if exists
    sample_doc = Path("sample_document.txt")
    if sample_doc.exists():
        print(f"\n[2] Indexing sample document...")
        with open(sample_doc, 'r', encoding='utf-8') as f:
            content = f.read()
        hrsc.insert(content)
        print("[OK] Document indexed")
    else:
        print(f"\n[2] No sample document found, skipping indexing")
    
    # Run sample queries
    print("\n[3] Running sample queries...")
    print("-"*80)
    
    queries = [
        "What is HiRAG?",
        "What is HiRAG?",  # Repeat to demonstrate caching
        "How does hierarchical retrieval work?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        answer = hrsc.query(query, mode="naive")  # Use naive mode since hierarchical is disabled
        print(f"Answer: {answer[:200]}...")
    
    # Show metrics
    print("\n[4] Performance Metrics:")
    print("-"*80)
    hrsc.print_metrics()
    
    print("\n" + "="*80)
    print("QUICK START COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  - Run 'python hrsc_demo.py --help' for more options")
    print("  - Run 'python hrsc_demo.py --compare --query \"What is HiRAG?\"' for comparison")
    print("  - Run 'python benchmark_hrsc.py' for full performance evaluation")
    print("\n")


if __name__ == "__main__":
    main()
