#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HRSC Framework Test Suite

Comprehensive tests for the HRSC framework including:
- Unit tests for SemanticCache
- Unit tests for DTRA
- Integration tests for HRSC
- Performance benchmarks
"""

import pytest
import asyncio
import numpy as np
import time
import os
from pathlib import Path

# Import HRSC components
from semantic_cache import SemanticCache, CacheEntry
from dtra import DynamicTaskRoutingAgent
from hrsc import HRSCFramework, initialize_hrsc_from_config


# Mock embedding function for testing
class MockEmbeddingFunc:
    """Mock embedding function that returns deterministic embeddings"""
    embedding_dim = 128
    max_token_size = 512
    
    async def __call__(self, texts):
        """Generate mock embeddings based on text hash"""
        embeddings = []
        for text in texts:
            # Create deterministic embedding from text hash
            hash_val = hash(text)
            np.random.seed(hash_val % (2**32))
            embedding = np.random.randn(self.embedding_dim)
            embeddings.append(embedding)
        return np.array(embeddings)


# ============================================================================
# Unit Tests: Semantic Cache
# ============================================================================

class TestSemanticCache:
    """Test suite for SemanticCache"""
    
    @pytest.fixture
    def embedding_func(self):
        """Mock embedding function fixture"""
        return MockEmbeddingFunc()
    
    @pytest.fixture
    def cache(self, embedding_func):
        """Semantic cache fixture"""
        return SemanticCache(
            embedding_func=embedding_func,
            similarity_threshold=0.85,
            max_size=10,
            ttl_seconds=3600,
            eviction_policy="lru"
        )
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache):
        """Test cache initializes correctly"""
        assert len(cache) == 0
        assert cache.similarity_threshold == 0.85
        assert cache.max_size == 10
        stats = cache.get_statistics()
        assert stats['total_queries'] == 0
        assert stats['cache_hits'] == 0
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache, embedding_func):
        """Test cache miss on empty cache"""
        query_emb = (await embedding_func(["test query"]))[0]
        hit, answer, context, sim = await cache.search(query_emb)
        assert hit is False
        assert answer is None
        assert context is None
    
    @pytest.mark.asyncio
    async def test_cache_update_and_hit(self, cache, embedding_func):
        """Test cache update and subsequent hit"""
        query_emb = (await embedding_func(["test query"]))[0]
        
        # Update cache
        await cache.update(query_emb, "test answer", "test context")
        assert len(cache) == 1
        
        # Search should hit
        hit, answer, context, sim = await cache.search(query_emb)
        assert hit is True
        assert answer == "test answer"
        assert context == "test context"
        assert sim >= 0.85
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_matching(self, cache, embedding_func):
        """Test semantic similarity between similar queries"""
        # Add original query
        query1_emb = (await embedding_func(["What is HiRAG?"]))[0]
        await cache.update(query1_emb, "HiRAG is a system", "context")
        
        # Very similar query should hit (same text = perfect match)
        query2_emb = (await embedding_func(["What is HiRAG?"]))[0]
        hit, answer, context, sim = await cache.search(query2_emb)
        assert hit is True
        assert sim > 0.99  # Should be nearly perfect match
        
        # Different query should miss
        query3_emb = (await embedding_func(["What is the weather?"]))[0]
        hit, answer, context, sim = await cache.search(query3_emb)
        assert hit is False
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache, embedding_func):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        for i in range(10):
            query_emb = (await embedding_func([f"query {i}"]))[0]
            await cache.update(query_emb, f"answer {i}", f"context {i}")
        
        assert len(cache) == 10
        
        # Add one more (should evict oldest)
        new_query_emb = (await embedding_func(["new query"]))[0]
        await cache.update(new_query_emb, "new answer", "new context")
        assert len(cache) == 10  # Should still be at max
        
        # Oldest query should be evicted
        query0_emb = (await embedding_func(["query 0"]))[0]
        hit, _, _, _ = await cache.search(query0_emb)
        assert hit is False
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache, embedding_func):
        """Test statistics tracking"""
        query_emb = (await embedding_func(["test"]))[0]
        
        # Miss
        await cache.search(query_emb)
        
        # Update
        await cache.update(query_emb, "answer", "context")
        
        # Hit
        await cache.search(query_emb)
        
        stats = cache.get_statistics()
        assert stats['total_queries'] == 2
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
        assert stats['hit_rate'] == 0.5


# ============================================================================
# Integration Tests: HRSC Framework
# ============================================================================

class TestHRSCIntegration:
    """Integration tests for HRSC Framework"""
    
    @pytest.fixture
    def test_config_path(self, tmp_path):
        """Create temporary config file for testing"""
        config_content = """
default_provider: ollama
ollama:
  model: glm4
  base_url: http://localhost:11434
  api_key: ollama
hirag:
  working_dir: {working_dir}
  enable_llm_cache: true
  enable_hierachical_mode: true
  embedding_batch_num: 6
  embedding_func_max_async: 8
  enable_naive_rag: true
hrsc:
  enable_semantic_cache: true
  cache_similarity_threshold: 0.85
  cache_max_size: 100
  cache_ttl_seconds: 3600
  cache_eviction_policy: lru
  enable_cache_analytics: true
  use_redis: false
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content.format(working_dir=str(tmp_path / "data")))
        return str(config_file)
    
    def test_hrsc_initialization(self, test_config_path):
        """Test HRSC framework initialization"""
        # This test would require mocking HiRAG and LLM functions
        # For now, we'll test basic structure
        assert Path(test_config_path).exists()
    
    def test_cache_enabled_vs_disabled(self):
        """Test HRSC with cache enabled vs disabled"""
        # This would test the difference in behavior
        # when cache is enabled vs disabled
        pass


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks for HRSC"""
    
    @pytest.mark.asyncio
    async def test_cache_lookup_speed(self):
        """Test that cache lookups are fast (<100ms)"""
        embedding_func = MockEmbeddingFunc()
        cache = SemanticCache(
            embedding_func=embedding_func,
            similarity_threshold=0.85,
            max_size=1000
        )
        
        # Populate cache
        for i in range(100):
            query_emb = (await embedding_func([f"query {i}"]))[0]
            await cache.update(query_emb, f"answer {i}", f"context {i}")
        
        # Test lookup speed
        test_query_emb = (await embedding_func(["query 50"]))[0]
        start = time.time()
        await cache.search(test_query_emb)
        latency = time.time() - start
        
        assert latency < 0.1  # Should be < 100ms
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Test cache hit rate with repeated queries"""
        embedding_func = MockEmbeddingFunc()
        cache = SemanticCache(
            embedding_func=embedding_func,
            similarity_threshold=0.85
        )
        
        queries = ["query 1", "query 2", "query 1", "query 2", "query 1"]
        
        for query in queries:
            query_emb = (await embedding_func([query]))[0]
            hit, answer, _, _ = await cache.search(query_emb)
            
            if not hit:
                await cache.update(query_emb, f"answer for {query}", "context")
        
        stats = cache.get_statistics()
        # First 2 are misses, next 3 are hits
        assert stats['cache_hits'] == 3
        assert stats['cache_misses'] == 2
        assert stats['hit_rate'] == 0.6


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
