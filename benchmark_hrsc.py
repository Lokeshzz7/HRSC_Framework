#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HRSC Framework Benchmarking Script

Automated benchmarking for evaluating HRSC performance improvements over baseline HiRAG.
Generates comprehensive performance reports with metrics aligned to the HRSC paper.
"""

import argparse
import time
import logging
import statistics
from typing import List, Dict, Tuple
import json
from pathlib import Path

from hrsc import initialize_hrsc_from_config
from hi_rag_demo import initialize_hirag
from hirag import QueryParam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Automated benchmark runner for HRSC vs HiRAG comparison"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize benchmark runner"""
        self.config_path = config_path
        self.results = {
            "hirag_baseline": [],
            "hrsc_cache_cold": [],
            "hrsc_cache_warm": []
        }
    
    def generate_test_queries(
        self, 
        num_queries: int, 
        cache_hit_rate: float
    ) -> List[str]:
        """
        Generate test queries with controlled cache hit rate.
        
        Args:
            num_queries: Total number of queries
            cache_hit_rate: Target proportion of cache hits (0.0 to 1.0)
            
        Returns:
            List of query strings
        """
        base_queries = [
            "What is HiRAG and how does it work?",
            "Explain the hierarchical knowledge structure in HiRAG.",
            "What are the key differences between HiRAG and traditional RAG?",
            "How does the three-level retrieval process work?",
            "What are the advantages of hierarchical knowledge organization?",
            "Describe the local, global, and bridge knowledge layers.",
            "How does HiRAG reduce the knowledge gap?",
            "What is the role of entity extraction in HiRAG?",
            "How does HiRAG compare to GraphRAG?",
            "What are the evaluation metrics for HiRAG?",
        ]
        
        queries = []
        num_unique = int(num_queries * (1 - cache_hit_rate))
        num_repeated = num_queries - num_unique
        
        # Add unique queries
        for i in range(num_unique):
            base = base_queries[i % len(base_queries)]
            queries.append(f"{base} (variant {i // len(base_queries)})")
        
        # Add repeated queries (for cache hits)
        for i in range(num_repeated):
            queries.append(base_queries[i % len(base_queries)])
        
        return queries
    
    def benchmark_hirag_baseline(
        self, 
        queries: List[str],
        mode: str = "hi"
    ) -> Dict:
        """Benchmark baseline HiRAG (no caching)"""
        logger.info("Running HiRAG baseline benchmark...")
        
        hirag = initialize_hirag()
        latencies = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"HiRAG Query {i}/{len(queries)}")
            start = time.time()
            hirag.query(query, param=QueryParam(mode=mode))
            latency = time.time() - start
            latencies.append(latency)
        
        return {
            "latencies": latencies,
            "avg_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "total_time": sum(latencies)
        }
    
    def benchmark_hrsc(
        self,
        queries: List[str],
        mode: str = "hi"
    ) -> Dict:
        """Benchmark HRSC framework"""
        logger.info("Running HRSC benchmark...")
        
        hrsc = initialize_hrsc_from_config(self.config_path)
        hrsc.clear_cache()
        
        latencies = []
        paths = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"HRSC Query {i}/{len(queries)}")
            start = time.time()
            hrsc.query(query, mode=mode)
            latency = time.time() - start
            latencies.append(latency)
        
        metrics = hrsc.get_performance_metrics()
        
        return {
            "latencies": latencies,
            "avg_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "total_time": sum(latencies),
            "cache_hit_rate": metrics['dtra_metrics']['cache_hit_rate'],
            "fast_path_rate": metrics['dtra_metrics']['fast_path_rate'],
            "avg_fast_path_latency": metrics['dtra_metrics']['avg_fast_path_latency_ms'] / 1000,
            "avg_accuracy_path_latency": metrics['dtra_metrics']['avg_accuracy_path_latency_ms'] / 1000,
        }
    
    def run_comprehensive_benchmark(
        self,
        num_queries: int = 50,
        cache_hit_rates: List[float] = [0.0, 0.25, 0.5, 0.75],
        mode: str = "hi"
    ) -> Dict:
        """
        Run comprehensive benchmarks across different cache hit rates.
        
        Args:
            num_queries: Number of queries per benchmark
            cache_hit_rates: List of cache hit rates to test
            mode: Query mode
            
        Returns:
            Comprehensive results dictionary
        """
        results = {}
        
        for hit_rate in cache_hit_rates:
            logger.info(f"\n{'='*80}")
            logger.info(f"Benchmarking with {hit_rate:.0%} cache hit rate")
            logger.info(f"{'='*80}\n")
            
            queries = self.generate_test_queries(num_queries, hit_rate)
            
            # Run HRSC benchmark
            hrsc_results = self.benchmark_hrsc(queries, mode)
            
            results[f"hit_rate_{hit_rate:.2f}"] = {
                "cache_hit_rate": hit_rate,
                "hrsc": hrsc_results
            }
        
        # Run one baseline HiRAG benchmark (no caching, so hit rate doesn't matter)
        logger.info(f"\n{'='*80}")
        logger.info("Running baseline HiRAG benchmark (no caching)")
        logger.info(f"{'='*80}\n")
        
        baseline_queries = self.generate_test_queries(num_queries, 0.0)
        hirag_results = self.benchmark_hirag_baseline(baseline_queries, mode)
        results["hirag_baseline"] = hirag_results
        
        return results
    
    def print_report(self, results: Dict):
        """Print formatted benchmark report"""
        print("\n" + "="*80)
        print("HRSC BENCHMARK REPORT")
        print("="*80)
        
        # Baseline results
        if "hirag_baseline" in results:
            baseline = results["hirag_baseline"]
            print(f"\nHiRAG Baseline (No Caching):")
            print(f"  Avg Latency:    {baseline['avg_latency']*1000:.1f}ms")
            print(f"  Median Latency: {baseline['median_latency']*1000:.1f}ms")
            print(f"  Min Latency:    {baseline['min_latency']*1000:.1f}ms")
            print(f"  Max Latency:    {baseline['max_latency']*1000:.1f}ms")
            print(f"  Total Time:     {baseline['total_time']:.2f}s")
        
        # HRSC results for different hit rates
        print(f"\n{'-'*80}")
        print("HRSC Results by Cache Hit Rate:")
        print(f"{'-'*80}")
        
        baseline_avg = results.get("hirag_baseline", {}).get("avg_latency", 1.0)
        
        for key, data in sorted(results.items()):
            if key.startswith("hit_rate_"):
                hrsc = data["hrsc"]
                hit_rate = data["cache_hit_rate"]
                
                improvement = (baseline_avg - hrsc['avg_latency']) / baseline_avg * 100
                
                print(f"\nCache Hit Rate: {hit_rate:.0%}")
                print(f"  Avg Latency:         {hrsc['avg_latency']*1000:.1f}ms")
                print(f"  Fast Path Avg:       {hrsc['avg_fast_path_latency']*1000:.1f}ms")
                print(f"  Accuracy Path Avg:   {hrsc['avg_accuracy_path_latency']*1000:.1f}ms")
                print(f"  Actual Hit Rate:     {hrsc['cache_hit_rate']:.1%}")
                print(f"  Improvement vs Baseline: {improvement:.1f}%")
                print(f"  Total Time:          {hrsc['total_time']:.2f}s")
        
        print("\n" + "="*80)
        print("Paper Target: ≥90% latency reduction for cached queries")
        
        # Check if we meet the paper's target
        for key, data in results.items():
            if key == "hit_rate_0.50":  # 50% cache hit rate
                hrsc = data["hrsc"]
                if 'avg_fast_path_latency' in hrsc and hrsc['avg_fast_path_latency'] > 0:
                    cached_reduction = (baseline_avg - hrsc['avg_fast_path_latency']) / baseline_avg * 100
                    print(f"Achieved: {cached_reduction:.1f}% latency reduction for cached queries")
                    if cached_reduction >= 90:
                        print("✓ TARGET MET")
                    else:
                        print("✗ Target not met")
        
        print("="*80 + "\n")
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='HRSC Framework Benchmarking Tool'
    )
    parser.add_argument('--num-queries', type=int, default=50,
                       help='Number of queries per benchmark (default: 50)')
    parser.add_argument('--cache-hit-rates', type=float, nargs='+',
                       default=[0.0, 0.25, 0.5, 0.75],
                       help='Cache hit rates to test (default: 0.0 0.25 0.5 0.75)')
    parser.add_argument('--mode', type=str, default='naive',
                       choices=['hi', 'naive'],
                       help='Query mode (default: naive)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results (default: benchmark_results.json)')
    
    args = parser.parse_args()
    
    # Run benchmark
    runner = BenchmarkRunner(args.config)
    results = runner.run_comprehensive_benchmark(
        num_queries=args.num_queries,
        cache_hit_rates=args.cache_hit_rates,
        mode=args.mode
    )
    
    # Print report
    runner.print_report(results)
    
    # Save results
    runner.save_results(results, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
