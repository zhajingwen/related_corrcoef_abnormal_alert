"""
REST 客户端模块

基于 ccxt 库，提供带速率限制的 OHLCV 数据下载功能。
支持自动缓存到 SQLite，增量下载缺失数据。
"""

import time
import logging
from typing import Optional
import ccxt
import pandas as pd
from retry import retry

from .sqlite_cache import SQLiteCache

logger = logging.getLogger(__name__)


class RESTClient:
    """Hyperliquid REST API 客户端"""
    
    def __init__(
        self,
        exchange_name: str = "hyperliquid",
        timeout: int = 30000,
        cache: Optional[SQLiteCache] = None,
        enable_rate_limit: bool = True,
        rate_limit_ms: int = 500
    ):
        """
        初始化 REST 客户端
        
        Args:
            exchange_name: 交易所名称
            timeout: 请求超时时间（毫秒）
            cache: SQLite 缓存实例（可选）
            enable_rate_limit: 是否启用速率限制
            rate_limit_ms: 请求间隔（毫秒）
        """
        self.exchange_name = exchange_name
        self.exchange = getattr(ccxt, exchange_name)({
            "timeout": timeout,
            "enableRateLimit": enable_rate_limit,
            "rateLimit": rate_limit_ms,
        })
        self.cache = cache
        self.rate_limit_ms = rate_limit_ms
        
        logger.info(f"REST 客户端初始化 | 交易所: {exchange_name} | "
                    f"速率限制: {enable_rate_limit} | 间隔: {rate_limit_ms}ms")
    
    @staticmethod
    def timeframe_to_minutes(timeframe: str) -> int:
        """
        将 timeframe 字符串转换为分钟数
        
        支持的格式：m, h, d, w
        """
        unit_multipliers = {
            'm': 1,
            'h': 60,
            'd': 24 * 60,
            'w': 7 * 24 * 60,
        }
        
        unit = timeframe[-1].lower()
        if unit not in unit_multipliers:
            raise ValueError(f"不支持的 timeframe 格式: {timeframe}")
        
        value = int(timeframe[:-1])
        return value * unit_multipliers[unit]
    
    @staticmethod
    def period_to_bars(period: str, timeframe: str) -> int:
        """将时间周期转换为 K 线总条数"""
        days = int(period.rstrip('d'))
        timeframe_minutes = RESTClient.timeframe_to_minutes(timeframe)
        bars_per_day = int(24 * 60 / timeframe_minutes)
        return days * bars_per_day
    
    @retry(tries=10, delay=5, backoff=2, logger=logger)
    def _fetch_ohlcv_raw(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int = 1500
    ) -> list:
        """
        从交易所获取原始 OHLCV 数据（带重试）
        
        Args:
            symbol: 交易对
            timeframe: K 线周期
            since: 起始时间戳（毫秒）
            limit: 单次请求的最大条数
        
        Returns:
            OHLCV 数据列表 [[timestamp, open, high, low, close, volume], ...]
        """
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        period: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取 OHLCV 数据（支持缓存和增量下载）
        
        Args:
            symbol: 交易对，如 "BTC/USDC:USDC"
            timeframe: K 线周期，如 "5m"
            period: 数据周期，如 "60d"
            use_cache: 是否使用缓存
        
        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        target_bars = self.period_to_bars(period, timeframe)
        ms_per_bar = self.timeframe_to_minutes(timeframe) * 60 * 1000
        now_ms = self.exchange.milliseconds()
        since_ms = now_ms - target_bars * ms_per_bar
        
        # 检查缓存
        if use_cache and self.cache:
            cached_df = self._get_with_incremental_update(
                symbol, timeframe, since_ms, now_ms, ms_per_bar, target_bars
            )
            if cached_df is not None and len(cached_df) >= target_bars * 0.9:  # 允许 10% 的误差
                return self._process_dataframe(cached_df)
        
        # 全量下载
        logger.debug(f"全量下载 | {symbol} | {timeframe} | {period}")
        df = self._download_full(symbol, timeframe, since_ms, target_bars)
        
        # 保存到缓存
        if self.cache and not df.empty:
            self.cache.save_ohlcv(symbol, timeframe, df)
        
        return self._process_dataframe(df)
    
    def _get_with_incremental_update(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        now_ms: int,
        ms_per_bar: int,
        target_bars: int
    ) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据，并增量更新缺失部分
        
        Returns:
            更新后的 DataFrame，如果缓存不可用返回 None
        """
        # 获取缓存中的最新时间戳
        latest_cached = self.cache.get_latest_timestamp(symbol, timeframe)
        oldest_cached = self.cache.get_oldest_timestamp(symbol, timeframe)
        
        if latest_cached is None:
            logger.debug(f"缓存无数据 | {symbol} | {timeframe}")
            return None
        
        # 检查是否需要下载更早的数据
        if oldest_cached > since_ms:
            # 需要下载更早的历史数据
            logger.debug(f"下载历史数据 | {symbol} | {timeframe} | "
                        f"从 {since_ms} 到 {oldest_cached}")
            historical_df = self._download_range(symbol, timeframe, since_ms, oldest_cached - 1)
            if not historical_df.empty:
                self.cache.save_ohlcv(symbol, timeframe, historical_df)
        
        # 检查是否需要下载更新的数据
        if latest_cached < now_ms - ms_per_bar * 2:  # 允许 2 根 K 线的延迟
            # 需要下载最新数据
            new_since = latest_cached + 1
            logger.debug(f"增量更新 | {symbol} | {timeframe} | 从 {latest_cached}")
            new_df = self._download_range(symbol, timeframe, new_since, now_ms)
            if not new_df.empty:
                self.cache.save_ohlcv(symbol, timeframe, new_df)
        
        # 从缓存获取完整数据
        return self.cache.get_ohlcv(symbol, timeframe, since_ms=since_ms)
    
    def _download_full(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        target_bars: int
    ) -> pd.DataFrame:
        """全量下载 OHLCV 数据"""
        all_rows = []
        fetched = 0
        current_since = since_ms
        
        while True:
            try:
                ohlcv = self._fetch_ohlcv_raw(symbol, timeframe, current_since, limit=1500)
            except Exception as e:
                logger.error(f"下载失败 | {symbol} | {timeframe} | {e}")
                break
            
            if not ohlcv:
                break
            
            all_rows.extend(ohlcv)
            fetched += len(ohlcv)
            current_since = ohlcv[-1][0] + 1
            
            if len(ohlcv) < 1500 or fetched >= target_bars:
                break
            
            # 请求间隔
            time.sleep(self.rate_limit_ms / 1000)
        
        if not all_rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        return self._rows_to_dataframe(all_rows)
    
    def _download_range(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int
    ) -> pd.DataFrame:
        """下载指定时间范围的 OHLCV 数据"""
        all_rows = []
        current_since = since_ms
        
        while current_since < until_ms:
            try:
                ohlcv = self._fetch_ohlcv_raw(symbol, timeframe, current_since, limit=1500)
            except Exception as e:
                logger.error(f"下载范围失败 | {symbol} | {timeframe} | {e}")
                break
            
            if not ohlcv:
                break
            
            # 过滤超出范围的数据
            filtered = [row for row in ohlcv if row[0] <= until_ms]
            all_rows.extend(filtered)
            
            if len(ohlcv) < 1500 or ohlcv[-1][0] >= until_ms:
                break
            
            current_since = ohlcv[-1][0] + 1
            time.sleep(self.rate_limit_ms / 1000)
        
        if not all_rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        return self._rows_to_dataframe(all_rows)
    
    @staticmethod
    def _rows_to_dataframe(rows: list) -> pd.DataFrame:
        """将原始 OHLCV 行数据转换为 DataFrame"""
        df = pd.DataFrame(rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.set_index("Timestamp").sort_index()
        # 去除重复索引
        df = df[~df.index.duplicated(keep='last')]
        return df
    
    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """处理 DataFrame：添加 return 和 volume_usd 列"""
        if df.empty:
            df['return'] = pd.Series(dtype=float)
            df['volume_usd'] = pd.Series(dtype=float)
            return df
        
        df = df.copy()
        df['return'] = df['Close'].pct_change().fillna(0)
        df['volume_usd'] = df['Volume'] * df['Close']
        return df
    
    def load_markets(self) -> dict:
        """加载交易所市场信息"""
        return self.exchange.load_markets()
    
    def get_usdc_perpetuals(self) -> list[str]:
        """获取所有 USDC 永续合约交易对"""
        markets = self.load_markets()
        return [symbol for symbol in markets if '/USDC:USDC' in symbol]
    
    def milliseconds(self) -> int:
        """获取当前时间戳（毫秒）"""
        return self.exchange.milliseconds()

