"""
Async utilities for parallel processing.
"""
import asyncio
from typing import List, Callable, TypeVar, Any, Awaitable

T = TypeVar('T')
R = TypeVar('R')

async def process_batch_async(
    items: List[T], 
    process_func: Callable[[T], Awaitable[R]], 
    batch_size: int = 10
) -> List[R]:
    """
    Process a list of items in batches asynchronously.
    
    Args:
        items: List of items to process
        process_func: Async function to process each item
        batch_size: Number of items to process concurrently
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_tasks = [process_func(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
    
    return results

async def map_async(
    items: List[T],
    map_func: Callable[[T], Awaitable[R]],
    max_concurrency: int = 10
) -> List[R]:
    """
    Map an async function over a list of items with limited concurrency.
    
    Args:
        items: List of items to process
        map_func: Async function to apply to each item
        max_concurrency: Maximum number of concurrent tasks
        
    Returns:
        List of mapped results
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(item: T) -> R:
        async with semaphore:
            return await map_func(item)
    
    return await asyncio.gather(*[process_with_semaphore(item) for item in items])

async def reduce_async(
    items: List[T],
    map_func: Callable[[T], Awaitable[R]],
    reduce_func: Callable[[List[R]], Awaitable[R]],
    max_concurrency: int = 10
) -> R:
    """
    Implement a map-reduce pattern with async functions.
    
    Args:
        items: List of items to process
        map_func: Async function to apply to each item
        reduce_func: Async function to reduce mapped results
        max_concurrency: Maximum number of concurrent tasks
        
    Returns:
        Reduced result
    """
    # Map phase
    mapped_results = await map_async(items, map_func, max_concurrency)
    
    # Reduce phase
    return await reduce_func(mapped_results)