#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Task Routing Agent (DTRA) for HRSC Framework

This module implements the intelligent query router that decides between
the fast cache path and the accuracy HiRAG path based on semantic similarity.

Based on Algorithm 1 from the HRSC paper.
"""

import time
import logging
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
from hirag import QueryParam

logger = logging.getLogger(__name__)


@dataclass
class RoutingMetrics:
    """Performance metrics for DTRA routing decisions"""
    total_queries: int = 0
    fast_path_queries: int = 0
    accuracy_path_queries: int = 0
    total_fast_path_latency: float = 0.0
    total_accuracy_path_latency: float = 0.0
    total_latency_saved: float = 0.0
    
    @property
    def fast_path_rate(self) -> float:
        """Probability of taking fast path (Pfast)"""
        if self.total_queries == 0:
            return 0.0
        return self.fast_path_queries / self.total_queries
    
    @property
    def accuracy_path_rate(self) -> float:
        """Probability of taking accuracy path (Paccuracy)"""
        return 1.0 - self.fast_path_rate
    
    @property
    def avg_fast_path_latency(self) -> float:
        """Average latency for fast path queries"""
        if self.fast_path_queries == 0:
            return 0.0
        return self.total_fast_path_latency / self.fast_path_queries
    
    @property
    def avg_accuracy_path_latency(self) -> float:
        """Average latency for accuracy path queries"""
        if self.accuracy_path_queries == 0:
            return 0.0
        return self.total_accuracy_path_latency / self.accuracy_path_queries
    
    @property
    def avg_overall_latency(self) -> float:
        """Average overall latency across all queries"""
        if self.total_queries == 0:
            return 0.0
        total_latency = self.total_fast_path_latency + self.total_accuracy_path_latency
        return total_latency / self.total_queries


class DynamicTaskRoutingAgent:
    """
    Dynamic Task Routing Agent (DTRA)
    
    Implements intelligent query routing based on semantic cache hits.
    Routes queries to either:
    - Fast Path: Semantic Cache (low latency ~50ms)
    - Accuracy Path: Full HiRAG retrieval (high latency ~1000ms)
    
    This is the core arbitration layer that achieves the latency reduction
    while preserving accuracy.
    """
    
    def __init__(
        self,
        semantic_cache,
        hirag_instance,
        embedding_func,
        enable_analytics: bool = True
    ):
        """
        Initialize the DTRA.
        
        Args:
            semantic_cache: SemanticCache instance
            hirag_instance: HiRAG instance for accuracy path
            embedding_func: Function to generate query embeddings
            enable_analytics: Whether to track detailed metrics (default: True)
        """
        self.semantic_cache = semantic_cache
        self.hirag = hirag_instance
        self.embedding_func = embedding_func
        self.enable_analytics = enable_analytics
        
        # Performance metrics
        self.metrics = RoutingMetrics()
        
        logger.info("DTRA initialized successfully")
    
    async def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Input query string
            
        Returns:
            Query embedding as numpy array
        """
        # The embedding function expects a list of texts
        embeddings = await self.embedding_func([query])
        return embeddings[0]  # Return first (and only) embedding
    
    async def route_query(
        self,
        query: str,
        mode: str = "hi",
        threshold: Optional[float] = None
    ) -> Tuple[str, float, str, Dict]:
        """
        Route a query to either fast cache path or accuracy HiRAG path.
        
        This implements Algorithm 1 from the HRSC paper:
        1. Embed incoming query (Ein)
        2. Search semantic cache
        3. If Sim(Ein, Ecached) >= τ: Fast Path (cache hit)
        4. Else: Accuracy Path (HiRAG retrieval) + Cache Update
        
        Args:
            query: Input query string
            mode: Query mode for HiRAG (default: "hi")
            threshold: Optional override for similarity threshold
            
        Returns:
            Tuple of (answer, latency, path_taken, metadata)
            - answer: Generated or cached answer
            - latency: Query processing time in seconds
            - path_taken: "fast" or "accuracy"
            - metadata: Additional information about the routing decision
        """
        start_time = time.time()
        
        # Step 1: Embed the query
        logger.info(f"DTRA routing query: '{query[:50]}...'")
        query_embedding = await self._embed_query(query)
        
        # Step 2: Search semantic cache
        cache_hit, cached_answer, cached_context, similarity = await self.semantic_cache.search(
            query_embedding,
            threshold=threshold
        )
        
        # Update metrics
        self.metrics.total_queries += 1
        
        if cache_hit:
            # Step 3: FAST PATH - Cache Hit
            latency = time.time() - start_time
            self.metrics.fast_path_queries += 1
            self.metrics.total_fast_path_latency += latency
            
            # Estimate latency saved (assuming HiRAG would take ~1s)
            estimated_hirag_latency = 1.0  # Conservative estimate
            latency_saved = estimated_hirag_latency - latency
            self.metrics.total_latency_saved += latency_saved
            self.semantic_cache.stats.total_latency_saved += latency_saved
            
            logger.info(
                f"DTRA: FAST PATH taken (cache hit) - "
                f"latency={latency*1000:.1f}ms, similarity={similarity:.3f}"
            )
            
            metadata = {
                "path": "fast",
                "cache_hit": True,
                "similarity": similarity,
                "latency_saved_seconds": latency_saved,
                "cache_stats": self.semantic_cache.get_statistics()
            }
            
            return cached_answer, latency, "fast", metadata
        
        else:
            # Step 4: ACCURACY PATH - Cache Miss, invoke full HiRAG
            # Cache miss - use HiRAG for accurate retrieval
            logger.info(
                f"DTRA: ACCURACY PATH taken (cache miss) - "
                f"invoking HiRAG retrieval (similarity={similarity:.3f})"
            )
            
            # Invoke HiRAG for full hierarchical retrieval
            # Use async aquery to avoid event loop nesting issues
            hirag_start = time.time()
            answer = await self.hirag.aquery(query, param=QueryParam(mode=mode))
            hirag_latency = time.time() - hirag_start
            
            total_latency = time.time() - start_time
            
            self.metrics.accuracy_path_queries += 1
            self.metrics.total_accuracy_path_latency += total_latency
            
            # Step 5: Update cache with new result
            # Note: We don't have direct access to the retrieved context from HiRAG's query method
            # So we'll cache the answer with a placeholder context
            await self.semantic_cache.update(
                query_embedding=query_embedding,
                answer=answer,
                context="[HiRAG Retrieved Context]"
            )
            
            logger.info(
                f"DTRA: HiRAG retrieval completed - "
                f"latency={total_latency*1000:.1f}ms (HiRAG: {hirag_latency*1000:.1f}ms)"
            )
            
            metadata = {
                "path": "accuracy",
                "cache_hit": False,
                "similarity": similarity,
                "hirag_latency_seconds": hirag_latency,
                "cache_updated": True,
                "cache_stats": self.semantic_cache.get_statistics()
            }
            
            return answer, total_latency, "accuracy", metadata
    
    def get_metrics(self) -> Dict:
        """
        Get comprehensive DTRA performance metrics.
        
        Returns:
            Dictionary containing routing performance statistics
        """
        cache_stats = self.semantic_cache.get_statistics()
        
        return {
            # Routing metrics
            "total_queries": self.metrics.total_queries,
            "fast_path_queries": self.metrics.fast_path_queries,
            "accuracy_path_queries": self.metrics.accuracy_path_queries,
            "fast_path_rate": self.metrics.fast_path_rate,
            "accuracy_path_rate": self.metrics.accuracy_path_rate,
            
            # Latency metrics
            "avg_fast_path_latency_ms": self.metrics.avg_fast_path_latency * 1000,
            "avg_accuracy_path_latency_ms": self.metrics.avg_accuracy_path_latency * 1000,
            "avg_overall_latency_ms": self.metrics.avg_overall_latency * 1000,
            "total_latency_saved_seconds": self.metrics.total_latency_saved,
            
            # Cache metrics
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_size": cache_stats["current_cache_size"],
            "cache_utilization": cache_stats["cache_utilization"],
            "avg_cache_similarity": cache_stats["avg_similarity_on_hit"],
        }
    
    def print_metrics(self) -> None:
        """Print formatted performance metrics"""
        metrics = self.get_metrics()
        
        print("\n" + "="*80)
        print("DTRA PERFORMANCE METRICS")
        print("="*80)
        
        print(f"\nQuery Routing:")
        print(f"  Total Queries:        {metrics['total_queries']}")
        print(f"  Fast Path Queries:    {metrics['fast_path_queries']} ({metrics['fast_path_rate']:.1%})")
        print(f"  Accuracy Path Queries: {metrics['accuracy_path_queries']} ({metrics['accuracy_path_rate']:.1%})")
        
        print(f"\nLatency Performance:")
        print(f"  Avg Fast Path:        {metrics['avg_fast_path_latency_ms']:.1f} ms")
        print(f"  Avg Accuracy Path:    {metrics['avg_accuracy_path_latency_ms']:.1f} ms")
        print(f"  Avg Overall:          {metrics['avg_overall_latency_ms']:.1f} ms")
        print(f"  Total Latency Saved:  {metrics['total_latency_saved_seconds']:.2f} seconds")
        
        print(f"\nCache Performance:")
        print(f"  Cache Hit Rate:       {metrics['cache_hit_rate']:.1%}")
        print(f"  Cache Size:           {metrics['cache_size']}")
        print(f"  Cache Utilization:    {metrics['cache_utilization']:.1%}")
        print(f"  Avg Similarity (hit): {metrics['avg_cache_similarity']:.3f}")
        
        # Calculate performance improvement
        if metrics['accuracy_path_queries'] > 0:
            baseline_latency = (
                metrics['total_queries'] * metrics['avg_accuracy_path_latency_ms']
            )
            actual_latency = metrics['total_queries'] * metrics['avg_overall_latency_ms']
            improvement = (baseline_latency - actual_latency) / baseline_latency
            
            print(f"\nPerformance Improvement:")
            print(f"  Latency Reduction:    {improvement:.1%}")
            print(f"  Baseline (all HiRAG): {baseline_latency:.1f} ms")
            print(f"  Actual (with HRSC):   {actual_latency:.1f} ms")
        
        print("="*80 + "\n")
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        self.metrics = RoutingMetrics()
        logger.info("DTRA metrics reset")
    
    def __repr__(self) -> str:
        """String representation of DTRA"""
        return (
            f"DTRA(total_queries={self.metrics.total_queries}, "
            f"fast_path_rate={self.metrics.fast_path_rate:.2%}, "
            f"avg_latency={self.metrics.avg_overall_latency*1000:.1f}ms)"
        )
