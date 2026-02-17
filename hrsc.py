#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HRSC Framework - Hierarchical Retrieval and Semantic Caching 

Main framework class that integrates: 
- Semantic Cache Layer 
- Dynamic Task Routing Agent (DTRA) 
- HiRAG hierarchical retrieval 

This implements the complete HRSC architecture as described in: 
"A Dynamic Task Routing Agent integrated with a Hierarchical Retrieval and 
Semantic Caching (HRSC) framework" 
"""

import os
import logging
import yaml
import asyncio
import numpy as np
from typing import Optional, Dict
from pathlib import Path

from hirag import HiRAG, QueryParam
from semantic_cache import SemanticCache
from dtra import DynamicTaskRoutingAgent

logger = logging.getLogger(__name__)


class HRSCFramework:
    """
    Hierarchical Retrieval and Semantic Caching (HRSC) Framework
    
    This is the main class that integrates all HRSC components:
    - Semantic Cache for low-latency query responses
    - DTRA for intelligent routing decisions
    - HiRAG for high-accuracy hierarchical retrieval
    
    The framework achieves ≥90% latency reduction for cached queries while
    preserving the contextual accuracy of HiRAG.
    """
    
    def __init__(
        self,
        working_dir: str,
        embedding_func,
        best_model_func,
        cheap_model_func,
        # HRSC-specific parameters
        enable_semantic_cache: bool = True,
        cache_similarity_threshold: float = 0.85,
        cache_max_size: int = 1000,
        cache_ttl_seconds: float = 3600,
        cache_eviction_policy: str = "lru",
        enable_cache_analytics: bool = True,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379",
        # HiRAG parameters
        enable_llm_cache: bool = True,
        enable_hierachical_mode: bool = True,
        embedding_batch_num: int = 6,
        embedding_func_max_async: int = 8,
        enable_naive_rag: bool = True,
    ):
        """
        Initialize the HRSC Framework.
        
        Args:
            working_dir: Directory for storing HiRAG data
            embedding_func: Async function to generate embeddings
            best_model_func: LLM function for best quality (e.g., DeepSeek)
            cheap_model_func: LLM function for cheaper operations (e.g., GLM)
            enable_semantic_cache: Enable/disable semantic caching
            cache_similarity_threshold: Minimum similarity for cache hit (default: 0.85)
            cache_max_size: Maximum cache entries (default: 1000)
            cache_ttl_seconds: Cache entry TTL (default: 3600)
            cache_eviction_policy: "lru" or "lfu" (default: "lru")
            enable_cache_analytics: Track detailed metrics
            use_redis: Use Redis backend for cache
            redis_url: Redis connection URL
            enable_llm_cache: Enable HiRAG's LLM cache
            enable_hierachical_mode: Enable HiRAG hierarchical mode
            embedding_batch_num: Batch size for embeddings
            embedding_func_max_async: Max async embedding calls
            enable_naive_rag: Enable naive RAG mode
        """
        self.working_dir = working_dir
        self.embedding_func = embedding_func
        self.enable_semantic_cache = enable_semantic_cache
        
        logger.info("Initializing HRSC Framework...")
        logger.info(f"Working directory: {working_dir}")
        logger.info(f"Semantic Cache: {'ENABLED' if enable_semantic_cache else 'DISABLED'}")
        
        # Create working directory
        os.makedirs(working_dir, exist_ok=True)
        
        # Initialize HiRAG (the accuracy backbone)
        logger.info("Initializing HiRAG backbone...")
        self.hirag = HiRAG(
            working_dir=working_dir,
            enable_llm_cache=enable_llm_cache,
            embedding_func=embedding_func,
            best_model_func=cheap_model_func,  # Use GLM for both to avoid DeepSeek API
            cheap_model_func=cheap_model_func,
            enable_hierachical_mode=enable_hierachical_mode,
            embedding_batch_num=embedding_batch_num,
            embedding_func_max_async=embedding_func_max_async,
            enable_naive_rag=enable_naive_rag
        )
        
        # Initialize Semantic Cache
        if enable_semantic_cache:
            logger.info("Initializing Semantic Cache Layer...")
            self.semantic_cache = SemanticCache(
                embedding_func=embedding_func,
                similarity_threshold=cache_similarity_threshold,
                max_size=cache_max_size,
                ttl_seconds=cache_ttl_seconds,
                eviction_policy=cache_eviction_policy,
                use_redis=use_redis,
                redis_url=redis_url
            )
            
            # Initialize DTRA
            logger.info("Initializing Dynamic Task Routing Agent...")
            self.dtra = DynamicTaskRoutingAgent(
                semantic_cache=self.semantic_cache,
                hirag_instance=self.hirag,
                embedding_func=embedding_func,
                enable_analytics=enable_cache_analytics
            )
        else:
            self.semantic_cache = None
            self.dtra = None
            logger.info("HRSC running in HiRAG-only mode (cache disabled)")
        
        logger.info("HRSC Framework initialization complete!")
    
    def insert(self, content: str) -> None:
        """
        Index a document into the HiRAG knowledge base.
        
        This delegates directly to HiRAG's insert method.
        
        Args:
            content: Document text to index
        """
        logger.info("Indexing document into HiRAG...")
        self.hirag.insert(content)
        logger.info("Document indexing complete")
    
    async def query_async(
        self, 
        query: str, 
        mode: str = "hi",
        bypass_cache: bool = False
    ) -> str:
        """
        Process a query using HRSC (async version).
        
        This is the main query entry point that leverages DTRA for routing.
        
        Args:
            query: Query string
            mode: HiRAG query mode ("hi", "naive", "hi_nobridge", etc.)
            bypass_cache: Force HiRAG path even if cache hit exists
            
        Returns:
            Answer string
        """
        if not self.enable_semantic_cache or bypass_cache:
            # Direct HiRAG query (no caching)
            logger.info("Querying HiRAG directly (cache disabled/bypassed)")
            return self.hirag.query(query, param=QueryParam(mode=mode))
        
        # Use DTRA for intelligent routing
        answer, latency, path, metadata = await self.dtra.route_query(query, mode=mode)
        
        logger.info(
            f"Query completed via {path.upper()} path in {latency*1000:.1f}ms"
        )
        
        return answer
    
    def query(
        self, 
        query: str, 
        mode: str = "hi",
        bypass_cache: bool = False
    ) -> str:
        """
        Process a query using HRSC (sync version).
        
        This is a synchronous wrapper around query_async for convenience.
        
        Args:
            query: Query string
            mode: HiRAG query mode ("hi", "naive", "hi_nobridge", etc.)
            bypass_cache: Force HiRAG path even if cache hit exists
            
        Returns:
            Answer string
        """
        # Run async query in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.query_async(query, mode=mode, bypass_cache=bypass_cache)
        )
    
    def get_performance_metrics(self) -> Dict:
        """
        Get comprehensive performance metrics from all components.
        
        Returns:
            Dictionary with DTRA and cache statistics
        """
        if not self.enable_semantic_cache:
            return {
                "semantic_cache_enabled": False,
                "message": "Semantic cache is disabled. No metrics available."
            }
        
        dtra_metrics = self.dtra.get_metrics()
        cache_stats = self.semantic_cache.get_statistics()
        
        return {
            "semantic_cache_enabled": True,
            "dtra_metrics": dtra_metrics,
            "cache_statistics": cache_stats,
        }
    
    def print_metrics(self) -> None:
        """Print formatted performance metrics"""
        if not self.enable_semantic_cache:
            print("\n" + "="*80)
            print("HRSC METRICS")
            print("="*80)
            print("Semantic Cache is DISABLED")
            print("Running in HiRAG-only mode")
            print("="*80 + "\n")
            return
        
        # Use DTRA's print method for comprehensive output
        self.dtra.print_metrics()
    
    def clear_cache(self) -> None:
        """Clear the semantic cache and reset metrics"""
        if self.enable_semantic_cache:
            asyncio.run(self.semantic_cache.clear())
            self.dtra.reset_metrics()
            logger.info("Cache and metrics cleared")
        else:
            logger.warning("Cannot clear cache: semantic cache is disabled")
    
    def __repr__(self) -> str:
        """String representation of HRSC"""
        if self.enable_semantic_cache:
            cache_info = f"cache={len(self.semantic_cache)}"
        else:
            cache_info = "cache=disabled"
        
        return f"HRSCFramework(working_dir={self.working_dir}, {cache_info})"


def initialize_hrsc_from_config(config_path: str = "config.yaml") -> HRSCFramework:
    """
    Initialize HRSC Framework from a YAML configuration file.
    
    This is a convenience function that loads configuration and sets up
    the embedding and LLM functions.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Initialized HRSCFramework instance
    """
    # Import the functions from hi_rag_demo
    from hi_rag_demo import (
        embedding_function,
        deepseek_llm_function,
        glm_llm_function
    )
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract HiRAG config
    hirag_config = config.get('hirag', {})
    hrsc_config = config.get('hrsc', {})
    
    # Resolve working directory - handle ${VAR:-default} syntax
    working_dir = hirag_config.get('working_dir', './data')
    
    # If it contains ${...} syntax (environment variable not resolved), use default
    if '${' in str(working_dir) or working_dir.startswith('/app'):
        working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        logger.info(f"Using default working directory: {working_dir}")

    
    # Initialize HRSC
    framework = HRSCFramework(
        working_dir=working_dir,
        embedding_func=embedding_function,
        best_model_func=deepseek_llm_function,
        cheap_model_func=glm_llm_function,
        # HRSC parameters
        enable_semantic_cache=hrsc_config.get('enable_semantic_cache', True),
        cache_similarity_threshold=hrsc_config.get('cache_similarity_threshold', 0.85),
        cache_max_size=hrsc_config.get('cache_max_size', 1000),
        cache_ttl_seconds=hrsc_config.get('cache_ttl_seconds', 3600),
        cache_eviction_policy=hrsc_config.get('cache_eviction_policy', 'lru'),
        enable_cache_analytics=hrsc_config.get('enable_cache_analytics', True),
        use_redis=hrsc_config.get('use_redis', False),
        redis_url=hrsc_config.get('redis_url', 'redis://localhost:6379'),
        # HiRAG parameters
        enable_llm_cache=hirag_config.get('enable_llm_cache', True),
        enable_hierachical_mode=hirag_config.get('enable_hierachical_mode', True),
        embedding_batch_num=hirag_config.get('embedding_batch_num', 6),
        embedding_func_max_async=hirag_config.get('embedding_func_max_async', 8),
        enable_naive_rag=hirag_config.get('enable_naive_rag', True),
    )
    
    return framework
