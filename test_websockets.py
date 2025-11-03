import asyncio
import websockets
import json
import aiohttp
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v3"
WS_BASE_URL = "ws://localhost:8000/ws"


class WebSocketTestClient:
    """WebSocket client for real-time data streaming"""
    
    async def subscribe_trades(self):
        """Subscribe to real-time trade feed"""
        uri = f"{WS_BASE_URL}/trades"
        print(f"\n{'='*70}")
        print(f"WebSocket: SUBSCRIBE TO TRADES")
        print(f"{'='*70}")
        print(f"Connecting to: {uri}")
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✓ Connected to trades feed")
                print(f"Listening for trades (30 seconds)...\n")
                
                start_time = datetime.now()
                trade_count = 0
                
                while True:
                    try:
                        # Set timeout to 30 seconds
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        trade = json.loads(message)
                        trade_count += 1
                        
                        print(f"TRADE #{trade_count}:")
                        print(f"  ID: {trade['trade_id']}")
                        print(f"  Symbol: {trade['symbol']}")
                        print(f"  Price: {trade['price']}")
                        print(f"  Quantity: {trade['quantity']}")
                        print(f"  Aggressor: {trade['aggressor_side'].upper()}")
                        print(f"  Maker Fee: {trade['maker_fee']}")
                        print(f"  Taker Fee: {trade['taker_fee']}")
                        print(f"  Timestamp: {trade['timestamp']}\n")
                        
                    except asyncio.TimeoutError:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print(f"✓ Timeout reached after {elapsed:.1f}s")
                        print(f"✓ Received {trade_count} trades")
                        break
                        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    async def subscribe_orderbook(self, symbol):
        """Subscribe to order book updates"""
        uri = f"{WS_BASE_URL}/orderbook/{symbol}"
        print(f"\n{'='*70}")
        print(f"WebSocket: SUBSCRIBE TO ORDER BOOK - {symbol}")
        print(f"{'='*70}")
        print(f"Connecting to: {uri}")
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✓ Connected to {symbol} order book feed")
                print(f"Listening for updates (30 seconds)...\n")
                
                start_time = datetime.now()
                update_count = 0
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        orderbook = json.loads(message)
                        update_count += 1
                        
                        print(f"ORDER BOOK UPDATE #{update_count}:")
                        print(f"  Symbol: {orderbook['symbol']}")
                        print(f"  Timestamp: {orderbook['timestamp']}")
                        print(f"  Top 3 BIDS:")
                        for i, bid in enumerate(orderbook['bids'][:3], 1):
                            print(f"    {i}. Price: {bid[0]}, Qty: {bid[1]}")
                        print(f"  Top 3 ASKS:")
                        for i, ask in enumerate(orderbook['asks'][:3], 1):
                            print(f"    {i}. Price: {ask[0]}, Qty: {ask[1]}")
                        print()
                        
                    except asyncio.TimeoutError:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print(f"✓ Timeout reached after {elapsed:.1f}s")
                        print(f"✓ Received {update_count} order book updates")
                        break
                        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    async def subscribe_bbo(self, symbol):
        """Subscribe to BBO (Best Bid/Offer) updates"""
        uri = f"{WS_BASE_URL}/bbo/{symbol}"
        print(f"\n{'='*70}")
        print(f"WebSocket: SUBSCRIBE TO BBO - {symbol}")
        print(f"{'='*70}")
        print(f"Connecting to: {uri}")
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✓ Connected to {symbol} BBO feed")
                print(f"Listening for updates (30 seconds)...\n")
                
                start_time = datetime.now()
                update_count = 0
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        bbo = json.loads(message)
                        update_count += 1
                        
                        print(f"BBO UPDATE #{update_count}:")
                        print(f"  Symbol: {bbo['symbol']}")
                        print(f"  Best Bid: {bbo['best_bid']} (Size: {bbo['bid_size']})")
                        print(f"  Best Ask: {bbo['best_ask']} (Size: {bbo['ask_size']})")
                        print(f"  Spread: {bbo['spread']}")
                        print(f"  Spread (bps): {bbo['spread_bps']}")
                        print(f"  Timestamp: {bbo['timestamp']}\n")
                        
                    except asyncio.TimeoutError:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print(f"✓ Timeout reached after {elapsed:.1f}s")
                        print(f"✓ Received {update_count} BBO updates")
                        break
                        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    async def subscribe_analytics(self, symbol):
        """Subscribe to market analytics updates"""
        uri = f"{WS_BASE_URL}/analytics/{symbol}"
        print(f"\n{'='*70}")
        print(f"WebSocket: SUBSCRIBE TO ANALYTICS - {symbol}")
        print(f"{'='*70}")
        print(f"Connecting to: {uri}")
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"✓ Connected to {symbol} analytics feed")
                print(f"Listening for updates (30 seconds)...\n")
                
                start_time = datetime.now()
                update_count = 0
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        metrics = json.loads(message)
                        update_count += 1
                        
                        print(f"ANALYTICS UPDATE #{update_count}:")
                        print(f"  Symbol: {metrics['symbol']}")
                        print(f"  Last Price: {metrics['last_price']}")
                        print(f"  Opening Price: {metrics['opening_price']}")
                        print(f"  High: {metrics['high_price']}")
                        print(f"  Low: {metrics['low_price']}")
                        print(f"  Volume: {metrics['volume']}")
                        print(f"  VWAP: {metrics['vwap']}")
                        print(f"  TWAP: {metrics['twap']}")
                        print(f"  Trade Count: {metrics['trade_count']}")
                        print(f"  Volatility: {metrics['volatility']}")
                        print(f"  Timestamp: {metrics['timestamp']}\n")
                        
                    except asyncio.TimeoutError:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print(f"✓ Timeout reached after {elapsed:.1f}s")
                        print(f"✓ Received {update_count} analytics updates")
                        break
                        
        except Exception as e:
            print(f"✗ Error: {e}")


async def submit_test_orders():
    """Submit test orders via REST API to trigger trades and WebSocket updates"""
    print(f"\n{'='*70}")
    print("SUBMITTING TEST ORDERS TO TRIGGER WebSocket EVENTS")
    print(f"{'='*70}\n")
    
    async with aiohttp.ClientSession() as session:
        # Submit first order
        order1 = {
            "symbol": "BTC-USDT",
            "order_type": "limit",
            "side": "buy",
            "quantity": "1.0",
            "price": "50000",
            "user_id": "user_1"
        }
        
        async with session.post(f"{BASE_URL}/orders", json=order1) as resp:
            result = await resp.json()
            print(f"✓ Submitted BUY order: {result.get('order_id')}")
        
        await asyncio.sleep(1)
        
        # Submit second order to trigger trade
        order2 = {
            "symbol": "BTC-USDT",
            "order_type": "limit",
            "side": "sell",
            "quantity": "0.5",
            "price": "50000",
            "user_id": "user_2"
        }
        
        async with session.post(f"{BASE_URL}/orders", json=order2) as resp:
            result = await resp.json()
            print(f"✓ Submitted SELL order: {result.get('order_id')}")
        
        await asyncio.sleep(1)
        
        # Submit market order
        order3 = {
            "symbol": "BTC-USDT",
            "order_type": "market",
            "side": "buy",
            "quantity": "0.3",
            "user_id": "user_3"
        }
        
        async with session.post(f"{BASE_URL}/orders", json=order3) as resp:
            result = await resp.json()
            print(f"✓ Submitted MARKET order: {result.get('order_id')}")
        
        await asyncio.sleep(1)
        
        # Submit ETH orders
        order4 = {
            "symbol": "ETH-USDT",
            "order_type": "limit",
            "side": "buy",
            "quantity": "5.0",
            "price": "3000",
            "user_id": "user_1"
        }
        
        async with session.post(f"{BASE_URL}/orders", json=order4) as resp:
            result = await resp.json()
            print(f"✓ Submitted ETH BUY order: {result.get('order_id')}")
        
        await asyncio.sleep(1)
        
        # Submit ETH sell to trigger trade
        order5 = {
            "symbol": "ETH-USDT",
            "order_type": "limit",
            "side": "sell",
            "quantity": "3.0",
            "price": "3000",
            "user_id": "user_2"
        }
        
        async with session.post(f"{BASE_URL}/orders", json=order5) as resp:
            result = await resp.json()
            print(f"✓ Submitted ETH SELL order: {result.get('order_id')}")
        
        print(f"\n✓ Test orders submitted. WebSocket events should appear...\n")


async def run_websocket_tests():
    """Run all WebSocket tests"""
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║         MATCHING ENGINE WebSocket - COMPREHENSIVE TEST SUITE      ║
║                         v3.0 - Real-time Streaming                 ║
╚════════════════════════════════════════════════════════════════════╝

Starting WebSocket tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

NOTE: WebSocket will listen for 30 seconds per subscription.
      Order submission will happen in the background to generate events.
    """)
    
    client = WebSocketTestClient()
    
    # Submit test orders in background
    asyncio.create_task(submit_test_orders())
    
    # Wait a bit for orders to be submitted
    await asyncio.sleep(2)
    
    # Test 1: Trade Feed
    print("\n" + "="*70)
    print("TEST 1: REAL-TIME TRADE FEED")
    print("="*70)
    await client.subscribe_trades()
    
    # Test 2: BTC Order Book
    print("\n" + "="*70)
    print("TEST 2: BTC ORDER BOOK UPDATES")
    print("="*70)
    await client.subscribe_orderbook("BTC-USDT")
    
    # Test 3: ETH Order Book
    print("\n" + "="*70)
    print("TEST 3: ETH ORDER BOOK UPDATES")
    print("="*70)
    await client.subscribe_orderbook("ETH-USDT")
    
    # Test 4: BTC BBO
    print("\n" + "="*70)
    print("TEST 4: BTC BEST BID/OFFER UPDATES")
    print("="*70)
    await client.subscribe_bbo("BTC-USDT")
    
    # Test 5: ETH BBO
    print("\n" + "="*70)
    print("TEST 5: ETH BEST BID/OFFER UPDATES")
    print("="*70)
    await client.subscribe_bbo("ETH-USDT")
    
    # Test 6: BTC Analytics
    print("\n" + "="*70)
    print("TEST 6: BTC MARKET ANALYTICS")
    print("="*70)
    await client.subscribe_analytics("BTC-USDT")
    
    # Test 7: ETH Analytics
    print("\n" + "="*70)
    print("TEST 7: ETH MARKET ANALYTICS")
    print("="*70)
    await client.subscribe_analytics("ETH-USDT")
    
    # Final Summary
    print("\n" + "="*70)
    print("WEBSOCKET TESTS COMPLETED!")
    print("="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


async def run_concurrent_websockets():
    """Run multiple WebSocket subscriptions concurrently"""
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║       MATCHING ENGINE - CONCURRENT WebSocket SUBSCRIPTIONS        ║
║                  Multiple feeds at the same time                    ║
╚════════════════════════════════════════════════════════════════════╝

Starting concurrent WebSocket tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    client = WebSocketTestClient()
    
    # Submit test orders
    asyncio.create_task(submit_test_orders())
    await asyncio.sleep(2)
    
    print("\nRunning all subscriptions concurrently for 20 seconds...\n")
    
    # Run multiple subscriptions concurrently
    tasks = [
        asyncio.create_task(client.subscribe_trades()),
        asyncio.create_task(client.subscribe_orderbook("BTC-USDT")),
        asyncio.create_task(client.subscribe_bbo("BTC-USDT")),
        asyncio.create_task(client.subscribe_analytics("BTC-USDT")),
    ]
    
    # Run for limited time
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=20.0)
    except asyncio.TimeoutError:
        print("\n✓ Concurrent test completed")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    import sys
    
    print("""
Choose test mode:
1. Sequential WebSocket tests (recommended for first time)
2. Concurrent WebSocket tests (advanced)

Usage: python test_websockets.py [1 or 2]
Default: 1
    """)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    if mode == "2":
        asyncio.run(run_concurrent_websockets())
    else:
        asyncio.run(run_websocket_tests())