#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HRSC Framework Demo Script

This script demonstrates the HRSC framework capabilities including:
- Side-by-side comparison with HiRAG
- Performance metrics tracking
- Interactive query mode
- Cache hit/miss visualization
"""

import argparse
import logging
import time
import os
from pathlib import Path
from hrsc import initialize_hrsc_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def index_document(framework, document_path: str) -> bool:
    """Index a document into HRSC"""
    logger.info(f"Indexing document: {document_path}")
    
    document_path = Path(document_path)
    if not document_path.exists():
        logger.error(f"Document not found: {document_path}")
        return False
    
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        framework.insert(content)
        logger.info("Document indexed successfully")
        return True
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return False


def run_single_query(framework, query: str, mode: str = "hi", show_metrics: bool = True):
    """Run a single query and display results"""
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    start_time = time.time()
    answer = framework.query(query, mode=mode)
    latency = time.time() - start_time
    
    print(f"\nANSWER:\n{answer}")
    print(f"\nLatency: {latency*1000:.1f}ms")
    print("="*80)
    
    if show_metrics:
        framework.print_metrics()


def run_comparison(framework, query: str, mode: str = "hi"):
    """Run query comparison: first time (HiRAG) vs second time (cache)"""
    print("\n" + "="*80)
    print("COMPARISON MODE: First Query vs Cached Query")
    print("="*80)
    
    # Clear cache to ensure fresh start
    framework.clear_cache()
    
    # First query (should go through HiRAG)
    print("\n[1] First Query (Expected: ACCURACY PATH via HiRAG)")
    print("-"*80)
    start_time = time.time()
    answer1 = framework.query(query, mode=mode)
    latency1 = time.time() - start_time
    print(f"Latency: {latency1*1000:.1f}ms")
    print(f"Answer: {answer1[:200]}...")
    
    # Second query (should hit cache)
    print("\n[2] Second Query (Expected: FAST PATH via Cache)")
    print("-"*80)
    start_time = time.time()
    answer2 = framework.query(query, mode=mode)
    latency2 = time.time() - start_time
    print(f"Latency: {latency2*1000:.1f}ms")
    print(f"Answer: {answer2[:200]}...")
    
    # Calculate improvement
    improvement = (latency1 - latency2) / latency1 * 100
    speedup = latency1 / latency2
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"First Query (HiRAG):  {latency1*1000:.1f}ms")
    print(f"Second Query (Cache): {latency2*1000:.1f}ms")
    print(f"Improvement:          {improvement:.1f}%")
    print(f"Speedup:              {speedup:.1f}x")
    print("="*80)
    
    # Show detailed metrics
    framework.print_metrics()


def run_benchmark(framework, num_queries: int = 50, cache_hit_rate: float = 0.5):
    """Run automated benchmark with simulated queries"""
    print("\n" + "="*80)
    print(f"BENCHMARK MODE: {num_queries} queries, target hit rate: {cache_hit_rate:.0%}")
    print("="*80)
    
    # Clear cache
    framework.clear_cache()
    
    # Sample queries (alternate between unique and repeated)
    base_queries = [
        "What is HiRAG?",
        "What are the key features?",
        "How does hierarchical retrieval work?",
        "What is the difference between naive and hierarchical RAG?",
        "Explain the three-level knowledge structure.",
    ]
    
    queries = []
    unique_count = int(num_queries * (1 - cache_hit_rate))
    repeated_count = num_queries - unique_count
    
    # Add unique queries
    for i in range(unique_count):
        queries.append(base_queries[i % len(base_queries)] + f" (variant {i})")
    
    # Add repeated queries (to trigger cache hits)
    for i in range(repeated_count):
        queries.append(base_queries[i % len(base_queries)])
    
    # Run queries
    print(f"\nRunning {len(queries)} queries...")
    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query[:50]}...", end=' ')
        start = time.time()
        framework.query(query, mode="naive")
        latency = time.time() - start
        print(f"{latency*1000:.1f}ms")
    
    # Show results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    framework.print_metrics()


def interactive_mode(framework, mode: str = "hi"):
    """Interactive query mode"""
    print("\n" + "="*80)
    print("HRSC INTERACTIVE MODE")
    print("="*80)
    print("Commands:")
    print("  - Type your query and press Enter")
    print("  - 'metrics' - Show performance metrics")
    print("  - 'clear' - Clear cache and metrics")
    print("  - 'exit' - Quit")
    print("="*80)
    
    while True:
        query = input("\n> ").strip()
        
        if not query:
            continue
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        if query.lower() == 'metrics':
            framework.print_metrics()
            continue
        
        if query.lower() == 'clear':
            framework.clear_cache()
            print("Cache and metrics cleared")
            continue
        
        # Process query
        start_time = time.time()
        answer = framework.query(query, mode=mode)
        latency = time.time() - start_time
        
        print(f"\nAnswer:\n{answer}")
        print(f"\n[Latency: {latency*1000:.1f}ms]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='HRSC Framework Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index and query
  python hrsc_demo.py --index sample_document.txt --query "What is HiRAG?"
  
  # Comparison mode
  python hrsc_demo.py --compare --query "What is HiRAG?"
  
  # Benchmark mode
  python hrsc_demo.py --benchmark --num-queries 50
  
  # Interactive mode
  python hrsc_demo.py --interactive
        """
    )
    
    parser.add_argument('--index', type=str, help='Path to document to index')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--mode', type=str, default='naive',
                       choices=['hi', 'naive', 'hi_nobridge', 'hi_local', 'hi_global', 'hi_bridge'],
                       help='Query mode (default: naive)')
    parser.add_argument('--show-metrics', action='store_true',
                       help='Show performance metrics after query')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison: first query vs cached query')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run automated benchmark')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='Number of queries for benchmark (default: 50)')
    parser.add_argument('--cache-hit-rate', type=float, default=0.5,
                       help='Target cache hit rate for benchmark (default: 0.5)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Initialize HRSC
    print("Initializing HRSC Framework...")
    framework = initialize_hrsc_from_config(args.config)
    print("HRSC initialized successfully!\n")
    
    # Index document if provided
    if args.index:
        success = index_document(framework, args.index)
        if not success:
            return 1
    
    # Determine mode
    if args.benchmark:
        run_benchmark(framework, args.num_queries, args.cache_hit_rate)
    elif args.compare:
        if not args.query:
            print("Error: --compare requires --query")
            return 1
        run_comparison(framework, args.query, args.mode)
    elif args.interactive:
        interactive_mode(framework, args.mode)
    elif args.query:
        run_single_query(framework, args.query, args.mode, args.show_metrics)
    else:
        # Default to interactive if nothing specified
        interactive_mode(framework, args.mode)
    
    return 0


if __name__ == "__main__":
    exit(main())
