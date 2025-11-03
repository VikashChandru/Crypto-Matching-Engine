import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8000/api/v3"

async def test_order():
    """Test submitting an order"""
    async with aiohttp.ClientSession() as session:
        order = {
            "symbol": "BTC-USDT",
            "order_type": "limit",
            "side": "buy",
            "quantity": "1.5",
            "price": "50000",
            "user_id": "user_1"
        }
        
        async with session.post(f'{BASE_URL}/orders', json=order) as resp:
            result = await resp.json()
            print("Order Response:")
            print(json.dumps(result, indent=2))
            return result.get('order_id')

async def check_for_matches(order_id):
    """Check if order was matched"""
    if not order_id:
        return
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{BASE_URL}/orders/{order_id}') as resp:
            order = await resp.json()
            
            filled_qty = float(order.get('filled_quantity', 0))
            remaining_qty = float(order.get('remaining_quantity', 0))
            total_qty = float(order.get('quantity', 0))
            
            print("\n" + "="*60)
            print("MATCH STATUS CHECK")
            print("="*60)
            print(f"Order ID: {order_id}")
            print(f"Status: {order.get('status').upper()}")
            print(f"Total Quantity: {total_qty}")
            print(f"Filled Quantity: {filled_qty}")
            print(f"Remaining Quantity: {remaining_qty}")
            print(f"Fill Percentage: {order.get('fill_percentage')}%")
            
            # Check for matches
            if filled_qty > 0:
                print(f"\n✓ MATCH FOUND!")
                print(f"  → {filled_qty} out of {total_qty} units were matched")
                print(f"  → {order.get('fill_percentage')}% of order filled")
                
                if remaining_qty > 0:
                    print(f"  → Remaining {remaining_qty} units still on order book")
            else:
                print(f"\n✗ NO MATCH YET")
                print(f"  → Order is still waiting on the order book")
                print(f"  → {total_qty} units are pending")

async def test_orderbook():
    """Test getting order book"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{BASE_URL}/orderbook/BTC-USDT?levels=10') as resp:
            result = await resp.json()
            print("\nOrder Book:")
            print(json.dumps(result, indent=2))
            
            # Check if there are bids and asks
            bids = result.get('bids', [])
            asks = result.get('asks', [])
            
            print("\n" + "="*60)
            print("ORDER BOOK ANALYSIS")
            print("="*60)
            print(f"Total Bid Levels: {len(bids)}")
            print(f"Total Ask Levels: {len(asks)}")
            
            if bids:
                print(f"Best Bid: {bids[0][0]} (Qty: {bids[0][1]})")
            if asks:
                print(f"Best Ask: {asks[0][0]} (Qty: {asks[0][1]})")

async def test_bbo():
    """Test getting BBO"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{BASE_URL}/bbo/BTC-USDT') as resp:
            result = await resp.json()
            print("\nBBO (Best Bid/Offer):")
            print(json.dumps(result, indent=2))
            
            best_bid = result.get('best_bid')
            best_ask = result.get('best_ask')
            
            print("\n" + "="*60)
            print("SPREAD ANALYSIS")
            print("="*60)
            
            if best_bid and best_ask:
                print(f"Best Bid: {best_bid}")
                print(f"Best Ask: {best_ask}")
                print(f"Spread: {result.get('spread')} (in bps: {result.get('spread_bps')})")
            else:
                print("✗ No market data available yet")

async def test_trades():
    """Test getting trades"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{BASE_URL}/trades?symbol=BTC-USDT&limit=10') as resp:
            result = await resp.json()
            print("\nRecent Trades:")
            print(json.dumps(result, indent=2))
            
            if isinstance(result, list) and len(result) > 0:
                print("\n" + "="*60)
                print("TRADE ACTIVITY")
                print("="*60)
                print(f"Total Trades: {len(result)}")
                
                for i, trade in enumerate(result[-3:], 1):  # Show last 3 trades
                    print(f"\nTrade #{i}:")
                    print(f"  Price: {trade['price']}")
                    print(f"  Quantity: {trade['quantity']}")
                    print(f"  Aggressor: {trade['aggressor_side'].upper()}")
            else:
                print("\n✗ No trades executed yet")

async def test_statistics():
    """Test getting statistics"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{BASE_URL}/statistics') as resp:
            result = await resp.json()
            print("\nStatistics:")
            print(json.dumps(result, indent=2))
            
            print("\n" + "="*60)
            print("ENGINE STATISTICS")
            print("="*60)
            print(f"Total Orders: {result.get('total_orders')}")
            print(f"Total Trades: {result.get('total_trades')}")
            print(f"Orders/sec: {result.get('orders_per_second'):.2f}")
            print(f"Trades/sec: {result.get('trades_per_second'):.2f}")

async def test_performance():
    """Test getting performance metrics"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{BASE_URL}/performance') as resp:
            result = await resp.json()
            print("\nPerformance Metrics:")
            print(json.dumps(result, indent=2))

async def test_positions(user_id):
    """Test getting user positions"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{BASE_URL}/positions/{user_id}') as resp:
            result = await resp.json()
            print(f"\nPositions for {user_id}:")
            print(json.dumps(result, indent=2))
            
            if result:
                print("\n" + "="*60)
                print("USER POSITIONS")
                print("="*60)
                for symbol, pos in result.items():
                    print(f"\n{symbol}:")
                    print(f"  Quantity: {pos['quantity']}")
                    print(f"  Entry Price: {pos['entry_price']}")
                    print(f"  Current Price: {pos['current_price']}")
                    print(f"  PnL: {pos['unrealized_pnl']} ({pos['pnl_percentage']}%)")
            else:
                print("✗ No positions yet")

async def submit_matching_order():
    """Submit a matching order to test trading"""
    async with aiohttp.ClientSession() as session:
        order = {
            "symbol": "BTC-USDT",
            "order_type": "limit",
            "side": "sell",
            "quantity": "0.5",
            "price": "50000",
            "user_id": "user_2"
        }
        
        print("\n" + "="*60)
        print("SUBMITTING MATCHING SELL ORDER")
        print("="*60)
        print("This should match with the previous buy order...")
        
        async with session.post(f'{BASE_URL}/orders', json=order) as resp:
            result = await resp.json()
            print("\nMatching Order Response:")
            print(json.dumps(result, indent=2))
            return result.get('order_id')

async def main():
    print("="*60)
    print("TESTING MATCHING ENGINE - WITH MATCH DETECTION")
    print("="*60)
    
    # Test 1: Submit buy order
    print("\n1. SUBMITTING BUY ORDER...")
    order_id = await test_order()
    
    # Check for matches on the buy order
    await check_for_matches(order_id)
    
    # Test 2: Get order book before selling
    print("\n2. GETTING ORDER BOOK BEFORE MATCH...")
    await test_orderbook()
    
    # Test 3: Get BBO
    print("\n3. GETTING BBO...")
    await test_bbo()
    
    # Test 4: Submit a matching sell order
    print("\n4. SUBMITTING MATCHING SELL ORDER...")
    await asyncio.sleep(1)
    match_order_id = await submit_matching_order()
    
    # Test 5: Check for matches after selling
    print("\n5. CHECKING FOR MATCHES AFTER TRADE...")
    await asyncio.sleep(1)
    await check_for_matches(order_id)
    
    # Test 6: Get updated order book
    print("\n6. GETTING UPDATED ORDER BOOK...")
    await test_orderbook()
    
    # Test 7: Get trades
    print("\n7. GETTING TRADES...")
    await test_trades()
    
    # Test 8: Get statistics
    print("\n8. GETTING STATISTICS...")
    await test_statistics()
    
    # Test 9: Get performance
    print("\n9. GETTING PERFORMANCE...")
    await test_performance()
    
    # Test 10: Get positions
    print("\n10. GETTING POSITIONS...")
    await test_positions("user_1")
    await test_positions("user_2")

if __name__ == "__main__":
    asyncio.run(main())