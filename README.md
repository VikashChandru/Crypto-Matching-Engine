# Crypto-Matching-Engine


**High-Performance Cryptocurrency Order Matching Engine**

This project implements a high-speed, production-grade order matching engine for cryptocurrency exchanges using **Python**, **FastAPI**, and **WebSockets**. It supports multiple order types, real-time analytics, risk management, and low-latency order execution.

---

### Overview

• Processes orders with microsecond-level latency using an asynchronous, thread-safe architecture
• Supports multiple trading pairs with independent order books
• Implements advanced order types such as **Market**, **Limit**, **IOC**, **FOK**, **Stop Loss**, **Stop Limit**, **Take Profit**, and **Iceberg Orders**
• Provides real-time market data through WebSocket streams (live trades, order book, BBO, analytics)
• Integrates risk management controls including position limits, order size checks, and daily loss caps
• Includes tiered fee structures with dynamic maker/taker fee adjustments
• Ensures persistence through periodic state snapshots and automatic recovery
• Exposes a complete REST API for order management, market data, and performance metrics

---

### Technologies Used

• Python 3.8+
• FastAPI
• Uvicorn
• Pydantic
• SortedContainers
• WebSockets

---

### How to Use

1. **Install Dependencies**

   ```bash
   pip install fastapi uvicorn pydantic sortedcontainers websockets
   ```

2. **Run the Engine**

   ```bash
   python matching_engine_v3.py
   ```

3. **Access API**

   * REST API: `http://localhost:8000/api/v3/`
   * WebSocket Streams: `ws://localhost:8000/ws/`

4. **Example Request**

   ```bash
   curl -X POST http://localhost:8000/api/v3/orders \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "BTC-USD",
       "order_type": "limit",
       "side": "buy",
       "quantity": "1.0",
       "price": "45000.00",
       "user_id": "trader_1"
     }'
   ```

The system will return order acceptance, track its status, and update real-time market data streams.

---

### Outputs

• Real-time trade feeds and order book updates
• Performance metrics (average latency, throughput, etc.)
• Market analytics including VWAP, TWAP, volatility, and spreads
• User-level P&L and position tracking

---

### Disclaimer

This project is for **educational and research purposes only**.
It is not intended for live financial trading without additional security, compliance, and infrastructure hardening.


