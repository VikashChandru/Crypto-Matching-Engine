import asyncio
import json
import logging
import time
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
from sortedcontainers import SortedDict
import uuid
from enum import Enum
from pathlib import Path
import statistics

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('matching_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions for SortedDict (must be at module level for pickling)
# ============================================================================

def bid_key_func(x):
    """Sorting key for bids (descending order)"""
    return -x


def ask_key_func(x):
    """Sorting key for asks (ascending order)"""
    return x


# ============================================================================

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"
    FOK = "fok"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    ICEBERG = "iceberg"
    VWAP = "vwap"
    TWAP = "twap"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PENDING = "pending"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    CIRCUIT_BREAK = "circuit_break"
    HALTED = "halted"


@dataclass
class Order:
    order_id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal]
    timestamp: datetime
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: Decimal = Decimal('0')
    stop_price: Optional[Decimal] = None
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    iceberg_qty: Optional[Decimal] = None
    algo_id: Optional[str] = None
    time_in_force: str = "GTC"
    post_only: bool = False
    reduce_only: bool = False
    last_update: datetime = field(default_factory=datetime.utcnow)

    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> Decimal:
        if self.quantity == 0:
            return Decimal('0')
        return (self.filled_quantity / self.quantity * 100).quantize(Decimal('0.01'))

    def to_dict(self):
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price is not None else None,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'filled_quantity': str(self.filled_quantity),
            'fill_percentage': str(self.fill_percentage),
            'client_id': self.client_id,
            'user_id': self.user_id,
        }


@dataclass
class Trade:
    trade_id: str
    symbol: str
    price: Decimal
    quantity: Decimal
    aggressor_side: OrderSide
    maker_order_id: str
    taker_order_id: str
    timestamp: datetime
    maker_fee: Decimal = Decimal('0')
    taker_fee: Decimal = Decimal('0')
    maker_user_id: Optional[str] = None
    taker_user_id: Optional[str] = None
    notional_value: Decimal = Decimal('0')

    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'price': str(self.price),
            'quantity': str(self.quantity),
            'aggressor_side': self.aggressor_side.value,
            'maker_order_id': self.maker_order_id,
            'taker_order_id': self.taker_order_id,
            'timestamp': self.timestamp.isoformat(),
            'maker_fee': str(self.maker_fee),
            'taker_fee': str(self.taker_fee),
            'notional_value': str(self.notional_value)
        }


@dataclass
class BBO:
    symbol: str
    best_bid: Optional[Decimal]
    best_ask: Optional[Decimal]
    bid_size: Decimal
    ask_size: Decimal
    spread: Optional[Decimal]
    spread_bps: Optional[Decimal]
    timestamp: datetime

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'best_bid': str(self.best_bid) if self.best_bid is not None else None,
            'best_ask': str(self.best_ask) if self.best_ask is not None else None,
            'bid_size': str(self.bid_size),
            'ask_size': str(self.ask_size),
            'spread': str(self.spread) if self.spread is not None else None,
            'spread_bps': str(self.spread_bps) if self.spread_bps is not None else None,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class UserPosition:
    user_id: str
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0')

    def to_dict(self):
        pnl_pct = (self.unrealized_pnl / (self.entry_price * self.quantity) * 100) if (self.entry_price != 0 and self.quantity != 0) else 0
        return {
            'symbol': self.symbol,
            'quantity': str(self.quantity),
            'entry_price': str(self.entry_price),
            'current_price': str(self.current_price),
            'unrealized_pnl': str(self.unrealized_pnl),
            'pnl_percentage': str(pnl_pct),
        }


@dataclass
class MarketMetrics:
    symbol: str
    timestamp: datetime
    last_price: Decimal
    opening_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    vwap: Decimal
    twap: Decimal
    trade_count: int
    volatility: Decimal

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'last_price': str(self.last_price),
            'opening_price': str(self.opening_price),
            'high_price': str(self.high_price),
            'low_price': str(self.low_price),
            'volume': str(self.volume),
            'vwap': str(self.vwap),
            'twap': str(self.twap),
            'trade_count': self.trade_count,
            'volatility': str(self.volatility),
        }


# ============================================================================

class FeeModel:
    """Volume-tiered fee model"""

    def __init__(self):
        self.volume_tiers = {
            Decimal('0'): (Decimal('0.1'), Decimal('0.2')),
            Decimal('100000'): (Decimal('0.08'), Decimal('0.15')),
            Decimal('500000'): (Decimal('0.06'), Decimal('0.12')),
            Decimal('1000000'): (Decimal('0.04'), Decimal('0.10')),
            Decimal('5000000'): (Decimal('0.02'), Decimal('0.05')),
        }
        self.user_volumes: Dict[str, Decimal] = defaultdict(Decimal)

    def get_user_fees(self, user_id: str) -> Tuple[Decimal, Decimal]:
        volume = self.user_volumes.get(user_id, Decimal('0'))
        for threshold in sorted(self.volume_tiers.keys(), reverse=True):
            if volume >= threshold:
                return self.volume_tiers[threshold]
        return self.volume_tiers[Decimal('0')]

    def calculate_maker_fee(self, user_id: str, notional_value: Decimal) -> Decimal:
        maker_fee_bps, _ = self.get_user_fees(user_id)
        return (notional_value * maker_fee_bps / Decimal('10000')).quantize(Decimal('0.00000001'))

    def calculate_taker_fee(self, user_id: str, notional_value: Decimal) -> Decimal:
        _, taker_fee_bps = self.get_user_fees(user_id)
        return (notional_value * taker_fee_bps / Decimal('10000')).quantize(Decimal('0.00000001'))

    def add_user_volume(self, user_id: str, volume: Decimal):
        self.user_volumes[user_id] += volume


# ============================================================================

class RiskManager:
    """Advanced risk management"""

    def __init__(self):
        self.user_limits: Dict[str, Dict[str, Decimal]] = {}
        self.user_positions: Dict[str, Dict[str, UserPosition]] = defaultdict(dict)
        self.daily_losses: Dict[str, Decimal] = defaultdict(Decimal)

    def set_user_limits(self, user_id: str, daily_limit: Decimal, position_limit: Decimal, order_limit: Decimal):
        self.user_limits[user_id] = {
            'daily_limit': daily_limit,
            'position_limit': position_limit,
            'order_limit': order_limit
        }

    def check_risk(self, user_id: str, symbol: str, quantity: Decimal, price: Decimal, order_type: str) -> Tuple[bool, Optional[str]]:
        if user_id in self.user_limits:
            limits = self.user_limits[user_id]
            if quantity > limits['order_limit']:
                return False, "Order size limit exceeded"
            current_pos = self.user_positions[user_id].get(symbol)
            if current_pos and abs(current_pos.quantity + quantity) > limits['position_limit']:
                return False, "Position limit exceeded"
        return True, None

    def update_position(self, user_id: str, symbol: str, quantity: Decimal, price: Decimal, trade_type: str):
        if symbol not in self.user_positions[user_id]:
            self.user_positions[user_id][symbol] = UserPosition(
                user_id=user_id,
                symbol=symbol,
                quantity=Decimal('0'),
                entry_price=price,
                current_price=price,
                unrealized_pnl=Decimal('0')
            )

        pos = self.user_positions[user_id][symbol]

        if trade_type == "buy":
            new_qty = pos.quantity + quantity
            if new_qty != 0:
                pos.entry_price = ((pos.entry_price * pos.quantity) + (price * quantity)) / new_qty
            pos.quantity = new_qty
        else:
            pos.quantity -= quantity

        pos.current_price = price
        pos.unrealized_pnl = pos.quantity * (price - pos.entry_price)


# ============================================================================

class OrderBook:
    """Advanced order book with analytics"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        # FIXED: Use named functions instead of lambdas for pickle compatibility
        self.bids: SortedDict = SortedDict(bid_key_func)
        self.asks: SortedDict = SortedDict(ask_key_func)
        self.order_map: Dict[str, tuple] = {}
        self.stop_orders: List[Order] = []

        self.trade_history: deque = deque(maxlen=10000)
        self.last_price: Optional[Decimal] = None
        self.opening_price: Optional[Decimal] = None
        self.high_price: Optional[Decimal] = None
        self.low_price: Optional[Decimal] = None
        self.volume: Decimal = Decimal('0')
        self.trade_count: int = 0
        self.vwap_sum: Decimal = Decimal('0')
        self.vwap_qty: Decimal = Decimal('0')

    def add_order(self, order: Order):
        price = order.price
        side = order.side
        if price is None:
            return
        if side == OrderSide.BUY:
            if price not in self.bids:
                self.bids[price] = deque()
            self.bids[price].append(order)
        else:
            if price not in self.asks:
                self.asks[price] = deque()
            self.asks[price].append(order)
        self.order_map[order.order_id] = (price, side)

    def remove_order(self, order_id: str) -> Optional[Order]:
        if order_id not in self.order_map:
            for i, order in enumerate(self.stop_orders):
                if order.order_id == order_id:
                    return self.stop_orders.pop(i)
            return None
        price, side = self.order_map[order_id]
        book = self.bids if side == OrderSide.BUY else self.asks
        if price in book:
            orders = book[price]
            for i, order in enumerate(orders):
                if order.order_id == order_id:
                    del orders[i]
                    if len(orders) == 0:
                        del book[price]
                    del self.order_map[order_id]
                    return order
        return None

    def add_stop_order(self, order: Order):
        self.stop_orders.append(order)

    def get_best_bid(self) -> Optional[Decimal]:
        return next(iter(self.bids), None)

    def get_best_ask(self) -> Optional[Decimal]:
        return next(iter(self.asks), None)

    def record_trade(self, price: Decimal, quantity: Decimal):
        self.trade_history.append((datetime.utcnow(), price, quantity))
        self.last_price = price
        self.volume += quantity
        self.trade_count += 1

        self.vwap_sum += price * quantity
        self.vwap_qty += quantity

        if self.opening_price is None:
            self.opening_price = price
        self.high_price = max(self.high_price or price, price)
        self.low_price = min(self.low_price or price, price)

    def get_vwap(self) -> Decimal:
        if self.vwap_qty == 0:
            return self.last_price or Decimal('0')
        return (self.vwap_sum / self.vwap_qty).quantize(Decimal('0.00000001'))

    def get_twap(self) -> Decimal:
        if len(self.trade_history) == 0:
            return self.last_price or Decimal('0')
        prices = [t[1] for t in self.trade_history]
        return (sum(prices) / len(prices)).quantize(Decimal('0.00000001'))

    def get_bbo(self) -> BBO:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        bid_size = sum(o.remaining_quantity for o in self.bids[best_bid]) if best_bid else Decimal('0')
        ask_size = sum(o.remaining_quantity for o in self.asks[best_ask]) if best_ask else Decimal('0')
        spread = None
        spread_bps = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            if best_bid > 0:
                spread_bps = (spread / best_bid * 10000).quantize(Decimal('0.01'))
        return BBO(
            symbol=self.symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=spread,
            spread_bps=spread_bps,
            timestamp=datetime.utcnow()
        )

    def get_depth(self, levels: int = 10) -> Dict:
        bids_depth = []
        asks_depth = []
        for i, (price, orders) in enumerate(self.bids.items()):
            if i >= levels:
                break
            total_qty = sum(o.remaining_quantity for o in orders)
            bids_depth.append([str(price), str(total_qty)])
        for i, (price, orders) in enumerate(self.asks.items()):
            if i >= levels:
                break
            total_qty = sum(o.remaining_quantity for o in orders)
            asks_depth.append([str(price), str(total_qty)])
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': self.symbol,
            'bids': bids_depth,
            'asks': asks_depth
        }

    def check_stop_triggers(self, current_price: Decimal) -> List[Order]:
        triggered = []
        remaining = []
        for order in self.stop_orders:
            should_trigger = False
            if order.order_type == OrderType.STOP_LOSS:
                if order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_trigger = True
                elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_trigger = True
            elif order.order_type == OrderType.TAKE_PROFIT:
                if order.side == OrderSide.SELL and current_price >= order.stop_price:
                    should_trigger = True
                elif order.side == OrderSide.BUY and current_price <= order.stop_price:
                    should_trigger = True
            if should_trigger:
                triggered.append(order)
            else:
                remaining.append(order)
        self.stop_orders = remaining
        return triggered

    def get_market_metrics(self) -> MarketMetrics:
        prices = [t[1] for t in list(self.trade_history)[-20:]] if self.trade_history else []
        volatility = Decimal('0')
        if len(prices) > 1:
            mean = sum(prices) / len(prices)
            variance = sum((p - mean) ** 2 for p in prices) / len(prices)
            volatility = variance.sqrt()
        return MarketMetrics(
            symbol=self.symbol,
            timestamp=datetime.utcnow(),
            last_price=self.last_price or Decimal('0'),
            opening_price=self.opening_price or Decimal('0'),
            high_price=self.high_price or Decimal('0'),
            low_price=self.low_price or Decimal('0'),
            volume=self.volume,
            vwap=self.get_vwap(),
            twap=self.get_twap(),
            trade_count=self.trade_count,
            volatility=volatility
        )


# ============================================================================

class PersistenceManager:
    """State persistence"""

    def __init__(self, data_dir: str = "./engine_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.snapshot_file = self.data_dir / "engine_snapshot.pkl"

    def save_snapshot(self, engine_state: Dict):
        try:
            with open(self.snapshot_file, 'wb') as f:
                pickle.dump(engine_state, f)
            logger.info("Snapshot saved successfully")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

    def load_snapshot(self) -> Optional[Dict]:
        try:
            if self.snapshot_file.exists():
                with open(self.snapshot_file, 'rb') as f:
                    state = pickle.load(f)
                logger.info("Snapshot loaded successfully")
                return state
            return None
        except Exception as e:
            logger.error(f"Error loading snapshot: {e}")
            return None


# ============================================================================

class MatchingEngineV3:
    """Matching Engine v3.0"""

    def __init__(self, enable_persistence: bool = True):
        self.order_books: Dict[str, OrderBook] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: deque = deque(maxlen=100000)

        self.fee_model = FeeModel()
        self.risk_manager = RiskManager()
        self.persistence = PersistenceManager() if enable_persistence else None

        self.trade_subscribers: Set = set()
        self.orderbook_subscribers: Dict[str, Set] = defaultdict(set)
        self.bbo_subscribers: Dict[str, Set] = defaultdict(set)
        self.analytics_subscribers: Dict[str, Set] = defaultdict(set)

        self.order_count = 0
        self.trade_count = 0
        self.start_time = time.time()
        self.latency_samples: deque = deque(maxlen=10000)

        self.algo_orders: Dict[str, Any] = {}

        if self.persistence:
            self._load_state()

        logger.info("Engine v3.0 initialized")

    def get_or_create_orderbook(self, symbol: str) -> OrderBook:
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        return self.order_books[symbol]

    async def submit_order(self, order_params: Dict) -> Dict:
        start_time = time.perf_counter_ns()

        try:
            order = self._create_order(order_params)
            if not order:
                return {'status': 'rejected', 'reason': 'Invalid parameters'}

            if order.user_id:
                risk_ok, risk_msg = self.risk_manager.check_risk(
                    order.user_id, order.symbol, order.quantity,
                    order.price or Decimal('0'), order.order_type.value
                )
                if not risk_ok:
                    return {'status': 'rejected', 'reason': risk_msg}

            self.orders[order.order_id] = order
            self.order_count += 1

            logger.info(f"Order submitted: {order.order_id}")

            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT]:
                order_book = self.get_or_create_orderbook(order.symbol)
                order.status = OrderStatus.PENDING
                order_book.add_stop_order(order)
                return {
                    'status': 'accepted',
                    'order_id': order.order_id,
                    'order_status': 'pending'
                }

            await self._match_order(order)

            await self._broadcast_bbo(order.symbol)
            await self._broadcast_orderbook(order.symbol)
            await self._broadcast_analytics(order.symbol)

            if self.persistence and self.order_count % 100 == 0:
                self._save_state()

            processing_time = time.perf_counter_ns() - start_time
            self._record_latency(processing_time)

            return {
                'status': 'accepted',
                'order_id': order.order_id,
                'order_status': order.status.value,
                'filled_quantity': str(order.filled_quantity),
                'remaining_quantity': str(order.remaining_quantity),
                'fill_percentage': str(order.fill_percentage)
            }

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return {'status': 'rejected', 'reason': str(e)}

    def _create_order(self, params: Dict) -> Optional[Order]:
        try:
            order_type = OrderType(params['order_type'])
            side = OrderSide(params['side'])
            quantity = Decimal(str(params['quantity']))
            price = Decimal(str(params['price'])) if params.get('price') else None
            stop_price = Decimal(str(params['stop_price'])) if params.get('stop_price') else None

            if quantity <= 0:
                return None

            if order_type in [OrderType.LIMIT, OrderType.IOC, OrderType.FOK, OrderType.STOP_LIMIT]:
                if not price or price <= 0:
                    return None

            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=params['symbol'],
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                timestamp=datetime.utcnow(),
                client_id=params.get('client_id'),
                user_id=params.get('user_id'),
                time_in_force=params.get('time_in_force', 'GTC'),
                post_only=params.get('post_only', False),
            )

            return order
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

    async def _match_order(self, order: Order):
        order_book = self.get_or_create_orderbook(order.symbol)

        if order.order_type == OrderType.MARKET:
            await self._match_market_order(order, order_book)
        elif order.order_type == OrderType.LIMIT:
            await self._match_limit_order(order, order_book)
        elif order.order_type == OrderType.IOC:
            await self._match_ioc_order(order, order_book)
        elif order.order_type == OrderType.FOK:
            await self._match_fok_order(order, order_book)
        elif order.order_type == OrderType.ICEBERG:
            await self._match_iceberg_order(order, order_book)
        elif order.order_type == OrderType.VWAP:
            await self._match_vwap_order(order, order_book)
        elif order.order_type == OrderType.TWAP:
            await self._match_twap_order(order, order_book)

        current_price = order_book.last_price
        if current_price is not None:
            triggered = order_book.check_stop_triggers(current_price)
            for stop_order in triggered:
                if stop_order.order_type == OrderType.STOP_LOSS:
                    stop_order.order_type = OrderType.MARKET
                elif stop_order.order_type in [OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT]:
                    stop_order.order_type = OrderType.LIMIT
                stop_order.status = OrderStatus.NEW
                await self._match_order(stop_order)

    async def _match_limit_order(self, order: Order, order_book: OrderBook):
        """Match limit"""
        opposite_book = order_book.asks if order.side == OrderSide.BUY else order_book.bids

        if not order.post_only:
            while order.remaining_quantity > 0 and len(opposite_book) > 0:
                best_price = next(iter(opposite_book), None)
                if best_price is None:
                    break

                if order.side == OrderSide.BUY:
                    if order.price is None or order.price < best_price:
                        break
                else:
                    if order.price is None or order.price > best_price:
                        break

                await self._execute_at_price_level(order, order_book, best_price)

        if order.remaining_quantity > 0:
            order.status = OrderStatus.NEW if order.filled_quantity == 0 else OrderStatus.PARTIALLY_FILLED
            order_book.add_order(order)
        else:
            order.status = OrderStatus.FILLED

    async def _match_ioc_order(self, order: Order, order_book: OrderBook):
        """Match IOC"""
        opposite_book = order_book.asks if order.side == OrderSide.BUY else order_book.bids

        while order.remaining_quantity > 0 and len(opposite_book) > 0:
            best_price = next(iter(opposite_book), None)
            if best_price is None:
                break

            if order.side == OrderSide.BUY:
                if order.price is not None and order.price < best_price:
                    break
            else:
                if order.price is not None and order.price > best_price:
                    break

            await self._execute_at_price_level(order, order_book, best_price)

        order.status = OrderStatus.FILLED if order.remaining_quantity == 0 else OrderStatus.CANCELLED

    async def _match_fok_order(self, order: Order, order_book: OrderBook):
        """Match FOK"""
        opposite_book = order_book.asks if order.side == OrderSide.BUY else order_book.bids

        available_quantity = Decimal('0')
        prices_to_fill = []

        for price, orders in opposite_book.items():
            if order.side == OrderSide.BUY:
                if order.price is not None and order.price < price:
                    break
            else:
                if order.price is not None and order.price > price:
                    break

            level_qty = sum(o.remaining_quantity for o in orders)
            available_quantity += level_qty
            prices_to_fill.append(price)

            if available_quantity >= order.quantity:
                break

        if available_quantity < order.quantity:
            order.status = OrderStatus.CANCELLED
            return

        for price in prices_to_fill:
            if order.remaining_quantity == 0:
                break
            await self._execute_at_price_level(order, order_book, price)

        order.status = OrderStatus.FILLED

    async def _match_iceberg_order(self, order: Order, order_book: OrderBook):
        if not order.iceberg_qty:
            order.iceberg_qty = order.quantity / 10

        visible_qty = min(order.iceberg_qty, order.remaining_quantity)

        visible_order = Order(
            order_id=f"{order.order_id}_visible",
            symbol=order.symbol,
            order_type=OrderType.LIMIT,
            side=order.side,
            quantity=visible_qty,
            price=order.price,
            timestamp=order.timestamp,
            user_id=order.user_id
        )

        await self._match_limit_order(visible_order, order_book)
        order.filled_quantity += visible_order.filled_quantity

        if order.remaining_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            order_book.add_order(order)
        else:
            order.status = OrderStatus.FILLED

    async def _match_vwap_order(self, order: Order, order_book: OrderBook):
        """Queue VWAP algo order (placeholder)"""
        algo_id = str(uuid.uuid4())
        order.algo_id = algo_id
        self.algo_orders[algo_id] = {
            'type': 'VWAP',
            'order': order,
            'target_vwap': order_book.get_vwap(),
            'start_time': datetime.utcnow()
        }
        order.status = OrderStatus.NEW
        logger.info(f"VWAP order queued: {algo_id}")

    async def _match_twap_order(self, order: Order, order_book: OrderBook):
        """Queue TWAP algo order (placeholder)"""
        algo_id = str(uuid.uuid4())
        order.algo_id = algo_id
        self.algo_orders[algo_id] = {
            'type': 'TWAP',
            'order': order,
            'target_twap': order_book.get_twap(),
            'start_time': datetime.utcnow()
        }
        order.status = OrderStatus.NEW
        logger.info(f"TWAP order queued: {algo_id}")

    async def _match_market_order(self, order: Order, order_book: OrderBook):
        opposite_book = order_book.asks if order.side == OrderSide.BUY else order_book.bids

        while order.remaining_quantity > 0 and len(opposite_book) > 0:
            best_price = next(iter(opposite_book), None)
            if best_price is None:
                break
            await self._execute_at_price_level(order, order_book, best_price)

        order.status = OrderStatus.FILLED if order.remaining_quantity == 0 else OrderStatus.CANCELLED

    async def _execute_at_price_level(self, taker_order: Order, order_book: OrderBook, price: Decimal):
        opposite_book = order_book.asks if taker_order.side == OrderSide.BUY else order_book.bids

        if price not in opposite_book:
            return

        orders_at_level = opposite_book[price]

        while len(orders_at_level) > 0 and taker_order.remaining_quantity > 0:
            maker_order = orders_at_level[0]

            fill_qty = min(taker_order.remaining_quantity, maker_order.remaining_quantity)
            notional_value = fill_qty * price

            maker_fee = self.fee_model.calculate_maker_fee(maker_order.user_id or "default", notional_value)
            taker_fee = self.fee_model.calculate_taker_fee(taker_order.user_id or "default", notional_value)

            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=taker_order.symbol,
                price=price,
                quantity=fill_qty,
                aggressor_side=taker_order.side,
                maker_order_id=maker_order.order_id,
                taker_order_id=taker_order.order_id,
                timestamp=datetime.utcnow(),
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                maker_user_id=maker_order.user_id,
                taker_user_id=taker_order.user_id,
                notional_value=notional_value
            )

            taker_order.filled_quantity += fill_qty
            maker_order.filled_quantity += fill_qty

            if maker_order.user_id:
                self.risk_manager.update_position(
                    maker_order.user_id, taker_order.symbol, fill_qty, price,
                    "sell" if maker_order.side == OrderSide.SELL else "buy"
                )
            if taker_order.user_id:
                self.risk_manager.update_position(
                    taker_order.user_id, taker_order.symbol, fill_qty, price,
                    "buy" if taker_order.side == OrderSide.BUY else "sell"
                )

            if maker_order.remaining_quantity == 0:
                maker_order.status = OrderStatus.FILLED
                orders_at_level.popleft()
                if maker_order.order_id in order_book.order_map:
                    del order_book.order_map[maker_order.order_id]
            else:
                maker_order.status = OrderStatus.PARTIALLY_FILLED

            self.trades.append(trade)
            self.trade_count += 1
            order_book.record_trade(price, fill_qty)

            if maker_order.user_id:
                self.fee_model.add_user_volume(maker_order.user_id, notional_value)
            if taker_order.user_id:
                self.fee_model.add_user_volume(taker_order.user_id, notional_value)

            await self._broadcast_trade(trade)

        if len(orders_at_level) == 0:
            try:
                del opposite_book[price]
            except KeyError:
                pass

    async def cancel_order(self, order_id: str) -> Dict:
        if order_id not in self.orders:
            return {'status': 'error', 'reason': 'Not found'}

        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return {'status': 'error', 'reason': 'Closed'}

        order_book = self.order_books.get(order.symbol)
        if order_book:
            order_book.remove_order(order_id)

        order.status = OrderStatus.CANCELLED

        await self._broadcast_bbo(order.symbol)
        await self._broadcast_orderbook(order.symbol)

        return {'status': 'success', 'order_id': order_id}

    def get_orderbook(self, symbol: str, levels: int = 10) -> Dict:
        order_book = self.order_books.get(symbol)
        if not order_book:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'bids': [],
                'asks': []
            }
        return order_book.get_depth(levels)

    def get_bbo(self, symbol: str) -> Dict:
        order_book = self.order_books.get(symbol)
        if not order_book:
            return {
                'symbol': symbol,
                'best_bid': None,
                'best_ask': None,
                'bid_size': '0',
                'ask_size': '0',
                'timestamp': datetime.utcnow().isoformat()
            }
        return order_book.get_bbo().to_dict()

    def get_market_metrics(self, symbol: str) -> Dict:
        order_book = self.order_books.get(symbol)
        if not order_book:
            return {'symbol': symbol, 'error': 'No data'}
        return order_book.get_market_metrics().to_dict()

    def get_order(self, order_id: str) -> Optional[Dict]:
        order = self.orders.get(order_id)
        return order.to_dict() if order else None

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        trades = list(self.trades)[-limit:]
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        return [t.to_dict() for t in trades]

    def get_user_positions(self, user_id: str) -> Dict:
        positions = self.risk_manager.user_positions.get(user_id, {})
        return {symbol: pos.to_dict() for symbol, pos in positions.items()}

    def get_statistics(self) -> Dict:
        elapsed = time.time() - self.start_time
        return {
            'total_orders': self.order_count,
            'total_trades': self.trade_count,
            'orders_per_second': self.order_count / elapsed if elapsed > 0 else 0,
            'trades_per_second': self.trade_count / elapsed if elapsed > 0 else 0,
            'uptime_seconds': elapsed,
            'active_symbols': len(self.order_books),
        }

    def get_performance_metrics(self) -> Dict:
        if not self.latency_samples:
            return {'message': 'No data'}
        latencies_us = [l / 1000 for l in list(self.latency_samples)]
        return {
            'avg_latency_us': statistics.mean(latencies_us),
            'median_latency_us': statistics.median(latencies_us),
            'min_latency_us': min(latencies_us),
            'max_latency_us': max(latencies_us),
            'p95_latency_us': sorted(latencies_us)[int(len(latencies_us) * 0.95)],
            'p99_latency_us': sorted(latencies_us)[int(len(latencies_us) * 0.99)],
        }

    async def _broadcast_trade(self, trade: Trade):
        for subscriber in list(self.trade_subscribers):
            try:
                await subscriber(trade.to_dict())
            except:
                self.trade_subscribers.discard(subscriber)

    async def _broadcast_bbo(self, symbol: str):
        bbo = self.get_bbo(symbol)
        for subscriber in list(self.bbo_subscribers[symbol]):
            try:
                await subscriber(bbo)
            except:
                self.bbo_subscribers[symbol].discard(subscriber)

    async def _broadcast_orderbook(self, symbol: str):
        orderbook = self.get_orderbook(symbol)
        for subscriber in list(self.orderbook_subscribers[symbol]):
            try:
                await subscriber(orderbook)
            except:
                self.orderbook_subscribers[symbol].discard(subscriber)

    async def _broadcast_analytics(self, symbol: str):
        metrics = self.get_market_metrics(symbol)
        for subscriber in list(self.analytics_subscribers[symbol]):
            try:
                await subscriber(metrics)
            except:
                self.analytics_subscribers[symbol].discard(subscriber)

    def _record_latency(self, latency_ns: int):
        self.latency_samples.append(latency_ns)

    def subscribe_trades(self, callback):
        self.trade_subscribers.add(callback)

    def subscribe_bbo(self, symbol: str, callback):
        self.bbo_subscribers[symbol].add(callback)

    def subscribe_orderbook(self, symbol: str, callback):
        self.orderbook_subscribers[symbol].add(callback)

    def subscribe_analytics(self, symbol: str, callback):
        self.analytics_subscribers[symbol].add(callback)

    def _save_state(self):
        if not self.persistence:
            return
        state = {
            'order_books': self.order_books,
            'orders': self.orders,
            'trades': list(self.trades)[-1000:],
            'order_count': self.order_count,
            'trade_count': self.trade_count
        }
        self.persistence.save_snapshot(state)

    def _load_state(self):
        if not self.persistence:
            return
        state = self.persistence.load_snapshot()
        if state:
            self.order_books = state.get('order_books', {})
            self.orders = state.get('orders', {})
            self.trades = deque(state.get('trades', []), maxlen=100000)
            self.order_count = state.get('order_count', 0)
            self.trade_count = state.get('trade_count', 0)
            logger.info("State restored from snapshot")

    def shutdown(self):
        logger.info("Shutting down matching engine...")
        self._save_state()
        logger.info("Shutdown complete")


# ============================================================================

class OrderRequest(BaseModel):
    symbol: str
    order_type: str
    side: str
    quantity: str
    price: Optional[str] = None
    stop_price: Optional[str] = None
    client_id: Optional[str] = None
    user_id: Optional[str] = None

    # FIXED: Use field_validator instead of deprecated @validator
    @field_validator('order_type')
    @classmethod
    def validate_order_type(cls, v):
        valid = ['market', 'limit', 'ioc', 'fok', 'stop_loss', 'stop_limit', 'take_profit', 'iceberg', 'vwap', 'twap']
        if v not in valid:
            raise ValueError('Invalid order type')
        return v

    @field_validator('side')
    @classmethod
    def validate_side(cls, v):
        if v not in ['buy', 'sell']:
            raise ValueError('Invalid side')
        return v


# ============================================================================

matching_engine = MatchingEngineV3(enable_persistence=True)

app = FastAPI(
    title="Matching Engine v3.0",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, ws: WebSocket, channel: str):
        await ws.accept()
        self.connections[channel].append(ws)

    def disconnect(self, ws: WebSocket, channel: str):
        if ws in self.connections[channel]:
            self.connections[channel].remove(ws)

    async def broadcast(self, msg: dict, channel: str):
        bad = []
        for conn in list(self.connections[channel]):
            try:
                await conn.send_json(msg)
            except:
                bad.append(conn)
        for conn in bad:
            self.disconnect(conn, channel)


ws_mgr = ConnectionManager()


# REST API
@app.post("/api/v3/orders")
async def submit_order(order: OrderRequest):
    result = await matching_engine.submit_order(order.dict())
    if result['status'] == 'rejected':
        raise HTTPException(status_code=400, detail=result.get('reason'))
    return result


@app.delete("/api/v3/orders/{order_id}")
async def cancel_order(order_id: str):
    result = await matching_engine.cancel_order(order_id)
    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['reason'])
    return result


@app.get("/api/v3/orders/{order_id}")
async def get_order(order_id: str):
    order = matching_engine.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.get("/api/v3/orderbook/{symbol}")
async def get_orderbook(symbol: str, levels: int = Query(10, ge=1, le=50)):
    return matching_engine.get_orderbook(symbol, levels)


@app.get("/api/v3/bbo/{symbol}")
async def get_bbo(symbol: str):
    return matching_engine.get_bbo(symbol)


@app.get("/api/v3/metrics/{symbol}")
async def get_metrics(symbol: str):
    return matching_engine.get_market_metrics(symbol)


@app.get("/api/v3/trades")
async def get_trades(symbol: Optional[str] = None, limit: int = Query(100, ge=1, le=1000)):
    return matching_engine.get_trades(symbol, limit)


@app.get("/api/v3/positions/{user_id}")
async def get_positions(user_id: str):
    return matching_engine.get_user_positions(user_id)


@app.get("/api/v3/statistics")
async def get_statistics():
    return matching_engine.get_statistics()


@app.get("/api/v3/performance")
async def get_performance():
    return matching_engine.get_performance_metrics()


@app.get("/api/v3/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


# WebSocket
@app.websocket("/ws/trades")
async def ws_trades(ws: WebSocket):
    await ws_mgr.connect(ws, "trades")

    async def callback(data):
        await ws_mgr.broadcast(data, "trades")

    matching_engine.subscribe_trades(callback)

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_mgr.disconnect(ws, "trades")


@app.websocket("/ws/orderbook/{symbol}")
async def ws_orderbook(ws: WebSocket, symbol: str):
    ch = f"ob_{symbol}"
    await ws_mgr.connect(ws, ch)

    async def callback(data):
        await ws_mgr.broadcast(data, ch)

    matching_engine.subscribe_orderbook(symbol, callback)

    try:
        await ws.send_json(matching_engine.get_orderbook(symbol))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_mgr.disconnect(ws, ch)


@app.websocket("/ws/bbo/{symbol}")
async def ws_bbo(ws: WebSocket, symbol: str):
    ch = f"bbo_{symbol}"
    await ws_mgr.connect(ws, ch)

    async def callback(data):
        await ws_mgr.broadcast(data, ch)

    matching_engine.subscribe_bbo(symbol, callback)

    try:
        await ws.send_json(matching_engine.get_bbo(symbol))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_mgr.disconnect(ws, ch)


@app.websocket("/ws/analytics/{symbol}")
async def ws_analytics(ws: WebSocket, symbol: str):
    ch = f"analytics_{symbol}"
    await ws_mgr.connect(ws, ch)

    async def callback(data):
        await ws_mgr.broadcast(data, ch)

    matching_engine.subscribe_analytics(symbol, callback)

    try:
        await ws.send_json(matching_engine.get_market_metrics(symbol))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_mgr.disconnect(ws, ch)


@app.on_event("shutdown")
async def shutdown_event():
    matching_engine.shutdown()


# ============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║   Cryptocurrency Matching Engine v3.0 - COMPLETE IMPLEMENTATION   ║
╚════════════════════════════════════════════════════════════════════╝
Starting server...
""")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )