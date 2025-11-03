import asyncio
import aiohttp
import time
import statistics
from datetime import datetime
from typing import List, Dict
import json

BASE_URL = "http://localhost:8000/api/v3"


class PerformanceBenchmark:
    """Load testing and performance benchmarking"""
    
    def __init__(self):
        self.results = {
            'order_submission': [],
            'order_cancellation': [],
            'get_orderbook': [],
            'get_bbo': [],
            'get_trades': [],
            'get_statistics': [],
            'market_orders': [],
            'limit_orders': [],
        }
        self.errors = []
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = None
    
    async def benchmark_order_submission(self, num_orders=100):
        """Benchmark order submission latency"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: ORDER SUBMISSION ({num_orders} orders)")
        print(f"{'='*70}")
        
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_orders):
                order = {
                    "symbol": "BTC-USDT",
                    "order_type": "limit",
                    "side": "buy" if i % 2 == 0 else "sell",
                    "quantity": str(1.0 + (i % 5)),
                    "price": str(50000 + (i % 1000)),
                    "user_id": f"user_{i % 10}"
                }
                
                start = time.perf_counter()
                try:
                    async with session.post(f"{BASE_URL}/orders", json=order) as resp:
                        await resp.json()
                        self.total_requests += 1
                    latency = (time.perf_counter() - start) * 1000  # Convert to ms
                    latencies.append(latency)
                    self.results['order_submission'].append(latency)
                except Exception as e:
                    self.total_errors += 1
                    self.errors.append(f"Order submission error: {e}")
                    latencies.append(None)
                
                if (i + 1) % 20 == 0:
                    print(f"  Submitted {i + 1}/{num_orders} orders...", end='\r')
        
        self._print_latency_stats("Order Submission", latencies)
        return latencies
    
    async def benchmark_limit_orders(self, num_orders=50):
        """Benchmark different limit order scenarios"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: LIMIT ORDERS ({num_orders} orders)")
        print(f"{'='*70}")
        
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_orders):
                order = {
                    "symbol": "BTC-USDT",
                    "order_type": "limit",
                    "side": "buy",
                    "quantity": "1.0",
                    "price": "50000",
                    "user_id": f"user_{i % 5}"
                }
                
                start = time.perf_counter()
                try:
                    async with session.post(f"{BASE_URL}/orders", json=order) as resp:
                        await resp.json()
                        self.total_requests += 1
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    self.results['limit_orders'].append(latency)
                except Exception as e:
                    self.total_errors += 1
                    latencies.append(None)
                
                if (i + 1) % 10 == 0:
                    print(f"  Tested {i + 1}/{num_orders} limit orders...", end='\r')
        
        self._print_latency_stats("Limit Orders", latencies)
        return latencies
    
    async def benchmark_market_orders(self, num_orders=50):
        """Benchmark market orders"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: MARKET ORDERS ({num_orders} orders)")
        print(f"{'='*70}")
        
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_orders):
                order = {
                    "symbol": "BTC-USDT",
                    "order_type": "market",
                    "side": "buy" if i % 2 == 0 else "sell",
                    "quantity": "0.5",
                    "user_id": f"user_{i % 5}"
                }
                
                start = time.perf_counter()
                try:
                    async with session.post(f"{BASE_URL}/orders", json=order) as resp:
                        await resp.json()
                        self.total_requests += 1
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    self.results['market_orders'].append(latency)
                except Exception as e:
                    self.total_errors += 1
                    latencies.append(None)
                
                if (i + 1) % 10 == 0:
                    print(f"  Tested {i + 1}/{num_orders} market orders...", end='\r')
        
        self._print_latency_stats("Market Orders", latencies)
        return latencies
    
    async def benchmark_get_orderbook(self, num_requests=50):
        """Benchmark order book retrieval"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: GET ORDER BOOK ({num_requests} requests)")
        print(f"{'='*70}")
        
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start = time.perf_counter()
                try:
                    async with session.get(f"{BASE_URL}/orderbook/BTC-USDT") as resp:
                        await resp.json()
                        self.total_requests += 1
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    self.results['get_orderbook'].append(latency)
                except Exception as e:
                    self.total_errors += 1
                    latencies.append(None)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_requests} requests...", end='\r')
        
        self._print_latency_stats("Get Order Book", latencies)
        return latencies
    
    async def benchmark_get_bbo(self, num_requests=50):
        """Benchmark BBO retrieval"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: GET BBO ({num_requests} requests)")
        print(f"{'='*70}")
        
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start = time.perf_counter()
                try:
                    async with session.get(f"{BASE_URL}/bbo/BTC-USDT") as resp:
                        await resp.json()
                        self.total_requests += 1
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    self.results['get_bbo'].append(latency)
                except Exception as e:
                    self.total_errors += 1
                    latencies.append(None)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_requests} requests...", end='\r')
        
        self._print_latency_stats("Get BBO", latencies)
        return latencies
    
    async def benchmark_get_trades(self, num_requests=50):
        """Benchmark trades retrieval"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: GET TRADES ({num_requests} requests)")
        print(f"{'='*70}")
        
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start = time.perf_counter()
                try:
                    async with session.get(f"{BASE_URL}/trades?symbol=BTC-USDT") as resp:
                        await resp.json()
                        self.total_requests += 1
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                    self.results['get_trades'].append(latency)
                except Exception as e:
                    self.total_errors += 1
                    latencies.append(None)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_requests} requests...", end='\r')
        
        self._print_latency_stats("Get Trades", latencies)
        return latencies
    
    async def benchmark_concurrent_requests(self, num_concurrent=50):
        """Benchmark concurrent requests"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: CONCURRENT REQUESTS ({num_concurrent} concurrent)")
        print(f"{'='*70}")
        
        async def make_request(session, index):
            order = {
                "symbol": "BTC-USDT",
                "order_type": "limit",
                "side": "buy" if index % 2 == 0 else "sell",
                "quantity": "0.5",
                "price": str(50000 + (index % 500)),
                "user_id": f"user_{index % 10}"
            }
            
            start = time.perf_counter()
            try:
                async with session.post(f"{BASE_URL}/orders", json=order) as resp:
                    await resp.json()
                    self.total_requests += 1
                return (time.perf_counter() - start) * 1000
            except Exception as e:
                self.total_errors += 1
                self.errors.append(f"Concurrent request error: {e}")
                return None
        
        latencies = []
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            latencies = [r for r in results if r is not None]
        
        self._print_latency_stats("Concurrent Requests", latencies)
        return latencies
    
    def _print_latency_stats(self, operation_name: str, latencies: List[float]):
        """Print latency statistics"""
        valid_latencies = [l for l in latencies if l is not None]
        
        if not valid_latencies:
            print(f"\n✗ No valid results for {operation_name}")
            return
        
        min_latency = min(valid_latencies)
        max_latency = max(valid_latencies)
        avg_latency = statistics.mean(valid_latencies)
        median_latency = statistics.median(valid_latencies)
        
        sorted_latencies = sorted(valid_latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        stdev = statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0
        
        print(f"\n{operation_name} - Latency Statistics:")
        print(f"  Min:     {min_latency:.2f} ms")
        print(f"  Max:     {max_latency:.2f} ms")
        print(f"  Mean:    {avg_latency:.2f} ms")
        print(f"  Median:  {median_latency:.2f} ms")
        print(f"  P50:     {p50:.2f} ms")
        print(f"  P95:     {p95:.2f} ms")
        print(f"  P99:     {p99:.2f} ms")
        print(f"  StdDev:  {stdev:.2f} ms")
        print(f"  Success: {len(valid_latencies)}/{len(latencies)}")
    
    async def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║           MATCHING ENGINE - PERFORMANCE BENCHMARK SUITE           ║
║                      Load Testing & Analysis                        ║
╚════════════════════════════════════════════════════════════════════╝

Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
        
        self.start_time = time.time()
        
        # Run benchmarks
        await self.benchmark_order_submission(num_orders=100)
        await self.benchmark_limit_orders(num_orders=50)
        await self.benchmark_market_orders(num_orders=50)
        await self.benchmark_concurrent_requests(num_concurrent=50)
        await self.benchmark_get_orderbook(num_requests=50)
        await self.benchmark_get_bbo(num_requests=50)
        await self.benchmark_get_trades(num_requests=50)
        
        # Print summary
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Total Requests: {self.total_requests}")
        print(f"Total Errors: {self.total_errors}")
        print(f"Total Time: {elapsed:.2f} seconds")
        print(f"Requests/sec: {self.total_requests / elapsed:.2f}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results
    
    def get_report_data(self) -> Dict:
        """Get data for report generation"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'results': {}
        }
        
        for operation, latencies in self.results.items():
            valid_latencies = [l for l in latencies if l is not None]
            
            if valid_latencies:
                sorted_latencies = sorted(valid_latencies)
                report_data['results'][operation] = {
                    'min': min(valid_latencies),
                    'max': max(valid_latencies),
                    'mean': statistics.mean(valid_latencies),
                    'median': statistics.median(valid_latencies),
                    'p50': sorted_latencies[int(len(sorted_latencies) * 0.50)],
                    'p95': sorted_latencies[int(len(sorted_latencies) * 0.95)],
                    'p99': sorted_latencies[int(len(sorted_latencies) * 0.99)],
                    'stdev': statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
                    'count': len(valid_latencies)
                }
        
        return report_data


async def main():
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # Save results to JSON for report generation
    report_data = benchmark.get_report_data()
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✓ Benchmark results saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())