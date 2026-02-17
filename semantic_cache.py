#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic Cache Layer for HRSC Framework

This module implements a high-speed semantic caching system using embedding-based
similarity matching. It provides LRU/LFU eviction policies with TTL support.

Based on the HRSC paper: "A Dynamic Task Routing Agent integrated with a
Hierarchical Retrieval and Semantic Caching (HRSC) framework"
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata"""
    query_embedding: np.ndarray
    answer: str
    context: str
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def is_expired(self, ttl: float) -> bool:
        """Check if entry has exceeded TTL"""
        return (time.time() - self.timestamp) > ttl


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_saved: float = 0.0
    avg_similarity_on_hit: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate


class SemanticCache:
    """
    Semantic Cache implementation using embedding-based similarity matching.
    
    Features:
    - Cosine similarity-based query matching
    - LRU (Least Recently Used) eviction with TTL
    - LFU (Least Frequently Used) tracking
    - In-memory storage with optional Redis backend
    - Comprehensive performance statistics
    """
    
    def __init__(
        self,
        embedding_func,
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
        ttl_seconds: float = 3600,
        eviction_policy: str = "lru",
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize the Semantic Cache.
        
        Args:
            embedding_func: Async function to generate embeddings
            similarity_threshold: Minimum cosine similarity for cache hit (default: 0.85)
            max_size: Maximum number of cache entries (default: 1000)
            ttl_seconds: Time-to-live for cache entries in seconds (default: 3600)
            eviction_policy: "lru" or "lfu" (default: "lru")
            use_redis: Whether to use Redis backend (default: False)
            redis_url: Redis connection URL (default: "redis://localhost:6379")
        """
        self.embedding_func = embedding_func
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        self.use_redis = use_redis
        self.redis_url = redis_url
        
        # In-memory storage: key = hash, value = CacheEntry
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = CacheStatistics()
        
        # Redis client (if enabled)
        self.redis_client = None
        if use_redis:
            try:
                import redis.asyncio as aioredis
                self.redis_client = aioredis.from_url(redis_url)
                logger.info(f"Redis cache enabled at {redis_url}")
            except ImportError:
                logger.warning("Redis not installed. Falling back to in-memory cache.")
                self.use_redis = False
        
        logger.info(
            f"Semantic Cache initialized: threshold={similarity_threshold}, "
            f"max_size={max_size}, ttl={ttl_seconds}s, policy={eviction_policy}"
        )
    
    def _compute_cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    async def search(
        self, 
        query_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Optional[str], float]:
        """
        Search for a semantically similar query in the cache.
        
        Args:
            query_embedding: Embedding of the incoming query
            threshold: Optional override for similarity threshold
            
        Returns:
            Tuple of (hit, answer, context, similarity_score)
            - hit: True if cache hit, False otherwise
            - answer: Cached answer if hit, None otherwise
            - context: Cached context if hit, None otherwise
            - similarity_score: Highest similarity score found
        """
        threshold = threshold or self.similarity_threshold
        self.stats.total_queries += 1
        
        # Clean expired entries first
        await self._evict_expired_entries()
        
        if len(self.cache) == 0:
            self.stats.cache_misses += 1
            logger.debug("Cache miss: empty cache")
            return False, None, None, 0.0
        
        # Search for most similar entry
        max_similarity = 0.0
        best_match_key = None
        
        for key, entry in self.cache.items():
            similarity = self._compute_cosine_similarity(
                query_embedding, 
                entry.query_embedding
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_key = key
        
        # Check if best match exceeds threshold
        if max_similarity >= threshold and best_match_key:
            # Cache hit!
            entry = self.cache[best_match_key]
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Update LRU order
            self.cache.move_to_end(best_match_key)
            
            self.stats.cache_hits += 1
            self.stats.avg_similarity_on_hit = (
                (self.stats.avg_similarity_on_hit * (self.stats.cache_hits - 1) + max_similarity)
                / self.stats.cache_hits
            )
            
            logger.info(
                f"Cache HIT: similarity={max_similarity:.3f}, "
                f"access_count={entry.access_count}"
            )
            
            return True, entry.answer, entry.context, max_similarity
        else:
            # Cache miss
            self.stats.cache_misses += 1
            logger.debug(
                f"Cache MISS: max_similarity={max_similarity:.3f} < threshold={threshold:.3f}"
            )
            return False, None, None, max_similarity
    
    async def update(
        self,
        query_embedding: np.ndarray,
        answer: str,
        context: str
    ) -> None:
        """
        Add or update a cache entry.
        
        Args:
            query_embedding: Embedding of the query
            answer: Generated answer
            context: Retrieved context used for generation
        """
        # Generate cache key from embedding hash
        cache_key = str(hash(query_embedding.tobytes()))
        
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            await self._evict_entry()
        
        # Create new entry
        entry = CacheEntry(
            query_embedding=query_embedding,
            answer=answer,
            context=context,
            timestamp=time.time()
        )
        
        # Add to cache
        self.cache[cache_key] = entry
        self.cache.move_to_end(cache_key)
        
        logger.debug(f"Cache updated: key={cache_key[:8]}..., cache_size={len(self.cache)}")
    
    async def _evict_entry(self) -> None:
        """Evict an entry based on the configured eviction policy"""
        if len(self.cache) == 0:
            return
        
        if self.eviction_policy == "lru":
            # Remove least recently used (first item in OrderedDict)
            key, _ = self.cache.popitem(last=False)
            logger.debug(f"Evicted LRU entry: {key[:8]}...")
            
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            min_access_count = float('inf')
            lfu_key = None
            
            for key, entry in self.cache.items():
                if entry.access_count < min_access_count:
                    min_access_count = entry.access_count
                    lfu_key = key
            
            if lfu_key:
                del self.cache[lfu_key]
                logger.debug(
                    f"Evicted LFU entry: {lfu_key[:8]}... "
                    f"(access_count={min_access_count})"
                )
    
    async def _evict_expired_entries(self) -> None:
        """Remove all expired entries from the cache"""
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired(self.ttl_seconds):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            logger.debug(f"Evicted expired entry: {key[:8]}...")
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary containing cache performance metrics
        """
        return {
            "total_queries": self.stats.total_queries,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate": self.stats.hit_rate,
            "miss_rate": self.stats.miss_rate,
            "avg_similarity_on_hit": self.stats.avg_similarity_on_hit,
            "current_cache_size": len(self.cache),
            "max_cache_size": self.max_size,
            "cache_utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            "total_latency_saved_seconds": self.stats.total_latency_saved,
        }
    
    async def clear(self) -> None:
        """Clear all cache entries and reset statistics"""
        self.cache.clear()
        self.stats = CacheStatistics()
        logger.info("Cache cleared")
    
    def __len__(self) -> int:
        """Return the number of entries in the cache"""
        return len(self.cache)
    
    def __repr__(self) -> str:
        """String representation of the cache"""
        return (
            f"SemanticCache(size={len(self.cache)}/{self.max_size}, "
            f"hit_rate={self.stats.hit_rate:.2%}, "
            f"threshold={self.similarity_threshold})"
        )
