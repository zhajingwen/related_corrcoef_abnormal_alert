"""
数据管理器模块

统一管理 WebSocket 实时数据和 REST 历史数据，智能选择数据源。
提供一致的数据访问接口，自动处理缓存和增量更新。
"""

import logging
from typing import Optional
from pathlib import Path
import pandas as pd

from .sqlite_cache import SQLiteCache
from .rest_client import RESTClient
from .websocket_client import WebSocketClient, HAS_HYPERLIQUID_SDK

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理器
    
    智能选择数据源：
    - 短期数据（1d）：优先 WebSocket 缓存
    - 长期数据（7d+）：REST API + SQLite 缓存
    """
    
    # 短期数据周期（优先使用 WebSocket）
    SHORT_PERIODS = ["1d"]
    
    def __init__(
        self,
        exchange_name: str = "hyperliquid",
        db_path: str = "hyperliquid_data.db",
        use_websocket: bool = True,
        testnet: bool = False
    ):
        """
        初始化数据管理器
        
        Args:
            exchange_name: 交易所名称
            db_path: SQLite 数据库路径
            use_websocket: 是否启用 WebSocket
            testnet: 是否使用测试网
        """
        self.exchange_name = exchange_name
        self.use_websocket = use_websocket and HAS_HYPERLIQUID_SDK
        
        # 初始化 SQLite 缓存
        self.cache = SQLiteCache(db_path)
        
        # 初始化 REST 客户端
        self.rest_client = RESTClient(
            exchange_name=exchange_name,
            cache=self.cache
        )
        
        # 初始化 WebSocket 客户端（如果启用）
        self.ws_client: Optional[WebSocketClient] = None
        if self.use_websocket:
            try:
                self.ws_client = WebSocketClient(testnet=testnet)
            except ImportError:
                logger.warning("WebSocket 客户端初始化失败，将只使用 REST API")
                self.use_websocket = False
        
        # BTC 数据缓存（用于分析）
        self._btc_cache: dict[tuple[str, str], pd.DataFrame] = {}
        
        logger.info(
            f"数据管理器初始化 | 交易所: {exchange_name} | "
            f"WebSocket: {self.use_websocket} | 数据库: {db_path}"
        )
    
    def initialize(self):
        """初始化：启动 WebSocket 连接"""
        if self.use_websocket and self.ws_client:
            self.ws_client.start()
            logger.info("WebSocket 已启动")
    
    def shutdown(self):
        """关闭：停止 WebSocket 连接"""
        if self.ws_client and self.ws_client.is_running:
            self.ws_client.stop()
            logger.info("WebSocket 已停止")
    
    def subscribe_realtime(self, coin: str, interval: str) -> bool:
        """
        订阅实时数据
        
        Args:
            coin: 币种符号，如 "BTC"
            interval: K 线周期，如 "5m"
        
        Returns:
            是否订阅成功
        """
        if not self.use_websocket or not self.ws_client:
            logger.warning("WebSocket 未启用，无法订阅实时数据")
            return False
        
        return self.ws_client.subscribe_candles(coin, interval)
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        period: str,
        prefer_realtime: bool = True
    ) -> pd.DataFrame:
        """
        获取 OHLCV 数据
        
        智能选择数据源：
        1. 短期数据（1d）+ 已订阅 → WebSocket 缓存
        2. SQLite 缓存 → 检查是否有足够数据
        3. REST API → 下载并缓存
        
        Args:
            symbol: 交易对，如 "BTC/USDC:USDC"
            timeframe: K 线周期，如 "5m"
            period: 数据周期，如 "60d"
            prefer_realtime: 是否优先使用实时数据
        
        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        # 计算需要的 K 线数量
        required_bars = self.rest_client.period_to_bars(period, timeframe)
        
        # 1. 短期数据：尝试 WebSocket 缓存
        if prefer_realtime and period in self.SHORT_PERIODS and self.use_websocket:
            coin = self._symbol_to_coin(symbol)
            if self.ws_client and self.ws_client.has_enough_data(coin, timeframe, required_bars):
                logger.debug(f"使用 WebSocket 缓存 | {symbol} | {timeframe} | {period}")
                df = self.ws_client.get_cached_dataframe(coin, timeframe, required_bars)
                return self._process_dataframe(df)
        
        # 2. 使用 REST 客户端（会自动检查 SQLite 缓存）
        logger.debug(f"使用 REST API | {symbol} | {timeframe} | {period}")
        return self.rest_client.fetch_ohlcv(symbol, timeframe, period)
    
    def get_btc_data(self, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        """
        获取 BTC 数据（带内存缓存）
        
        Args:
            timeframe: K 线周期
            period: 数据周期
        
        Returns:
            BTC 的 OHLCV 数据
        """
        cache_key = (timeframe, period)
        
        if cache_key in self._btc_cache:
            logger.debug(f"BTC 数据缓存命中 | {timeframe}/{period}")
            return self._btc_cache[cache_key].copy()
        
        btc_symbol = "BTC/USDC:USDC"
        try:
            df = self.get_ohlcv(btc_symbol, timeframe, period)
            if not df.empty:
                self._btc_cache[cache_key] = df
                return df.copy()
        except Exception as e:
            logger.error(f"获取 BTC 数据失败 | {timeframe}/{period} | {e}")
        
        return None
    
    def clear_btc_cache(self):
        """清除 BTC 内存缓存"""
        self._btc_cache.clear()
        logger.debug("BTC 内存缓存已清除")
    
    def get_usdc_perpetuals(self) -> list[str]:
        """获取所有 USDC 永续合约交易对"""
        return self.rest_client.get_usdc_perpetuals()
    
    def load_markets(self) -> dict:
        """加载市场信息"""
        return self.rest_client.load_markets()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        stats = {
            "sqlite": self.cache.get_cache_stats(),
            "websocket": None,
            "btc_memory_cache": list(self._btc_cache.keys())
        }
        
        if self.ws_client:
            stats["websocket"] = self.ws_client.get_subscription_status()
        
        return stats
    
    @staticmethod
    def _symbol_to_coin(symbol: str) -> str:
        """
        从交易对提取币种符号
        
        例如: "BTC/USDC:USDC" -> "BTC"
        """
        return symbol.split("/")[0]
    
    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """处理 DataFrame：添加 return 和 volume_usd 列"""
        if df.empty:
            df['return'] = pd.Series(dtype=float)
            df['volume_usd'] = pd.Series(dtype=float)
            return df
        
        df = df.copy()
        if 'return' not in df.columns:
            df['return'] = df['Close'].pct_change().fillna(0)
        if 'volume_usd' not in df.columns:
            df['volume_usd'] = df['Volume'] * df['Close']
        return df
    
    def prefetch_btc_data(self, timeframes: list[str], periods: list[str]):
        """
        预取 BTC 数据到缓存
        
        Args:
            timeframes: K 线周期列表
            periods: 数据周期列表
        """
        logger.info(f"预取 BTC 数据 | 周期: {timeframes} | 范围: {periods}")
        
        for timeframe in timeframes:
            for period in periods:
                try:
                    self.get_btc_data(timeframe, period)
                    logger.debug(f"预取完成 | BTC | {timeframe}/{period}")
                except Exception as e:
                    logger.error(f"预取失败 | BTC | {timeframe}/{period} | {e}")
    
    def prefetch_realtime_subscriptions(self, coins: list[str], intervals: list[str]):
        """
        预订阅实时数据
        
        Args:
            coins: 币种列表
            intervals: K 线周期列表
        """
        if not self.use_websocket:
            logger.warning("WebSocket 未启用，跳过实时订阅")
            return
        
        logger.info(f"预订阅实时数据 | 币种: {len(coins)} | 周期: {intervals}")
        
        for coin in coins:
            for interval in intervals:
                self.subscribe_realtime(coin, interval)

