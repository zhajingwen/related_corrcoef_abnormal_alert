"""
WebSocket 客户端模块

封装 Hyperliquid 官方 SDK 的 WebsocketManager，提供 K 线数据实时订阅功能。
支持多交易对订阅、内存缓存、自动重连。
"""

import logging
import asyncio
from collections import deque
from typing import Optional, Callable
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入官方 SDK
# 注意：如果当前目录有 hyperliquid.py 文件，会遮蔽安装的 hyperliquid 包
# 我们使用 importlib 来确保导入正确的包
HAS_HYPERLIQUID_SDK = False
Info = None
constants = None

def _try_import_hyperliquid():
    """尝试导入 hyperliquid SDK，处理本地文件遮蔽的问题"""
    global HAS_HYPERLIQUID_SDK, Info, constants
    
    try:
        import importlib.util
        import sys
        
        # 首先尝试直接导入
        try:
            from hyperliquid.info import Info as _Info
            from hyperliquid.utils import constants as _constants
            # 验证是否是正确的 SDK（检查是否有 WebsocketManager）
            from hyperliquid.websocket_manager import WebsocketManager
            Info = _Info
            constants = _constants
            HAS_HYPERLIQUID_SDK = True
            return True
        except (ImportError, AttributeError):
            pass
        
        # 如果失败，尝试从 site-packages 直接导入
        for path in sys.path:
            if 'site-packages' in path:
                spec = importlib.util.find_spec('hyperliquid', [path])
                if spec and spec.origin and 'site-packages' in spec.origin:
                    hyperliquid_pkg = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(hyperliquid_pkg)
                    
                    # 导入子模块
                    from hyperliquid.info import Info as _Info
                    from hyperliquid.utils import constants as _constants
                    Info = _Info
                    constants = _constants
                    HAS_HYPERLIQUID_SDK = True
                    return True
        
        return False
    except Exception as e:
        logger.debug(f"导入 hyperliquid SDK 失败: {e}")
        return False

# 执行导入
if not _try_import_hyperliquid():
    logger.warning(
        "hyperliquid SDK 未安装或导入失败，WebSocket 功能不可用。"
        "请运行: pip install hyperliquid-python-sdk"
    )


class WebSocketClient:
    """
    Hyperliquid WebSocket 客户端
    
    封装官方 SDK，提供 K 线数据实时订阅和内存缓存功能。
    """
    
    # 支持的 K 线周期
    SUPPORTED_INTERVALS = [
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    ]
    
    def __init__(self, max_cache_size: int = 1000, testnet: bool = False):
        """
        初始化 WebSocket 客户端
        
        Args:
            max_cache_size: 每个交易对/周期的最大缓存 K 线数量
            testnet: 是否使用测试网
        """
        if not HAS_HYPERLIQUID_SDK:
            raise ImportError("hyperliquid SDK 未安装，请运行: pip install hyperliquid")
        
        self.max_cache_size = max_cache_size
        self.testnet = testnet
        self.base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        
        # 数据缓存: {(coin, interval): deque([candle_data, ...])}
        self.data_cache: dict[tuple[str, str], deque] = {}
        
        # 订阅状态
        self.subscriptions: set[tuple[str, str]] = set()
        
        # Info 客户端（用于 WebSocket）
        self._info: Optional[Info] = None
        self._running = False
        
        # 回调函数
        self._callbacks: dict[tuple[str, str], list[Callable]] = {}
        
        logger.info(f"WebSocket 客户端初始化 | 测试网: {testnet} | 缓存大小: {max_cache_size}")
    
    def start(self):
        """启动 WebSocket 连接"""
        if self._running:
            logger.warning("WebSocket 已在运行中")
            return
        
        self._info = Info(self.base_url, skip_ws=False)
        self._running = True
        logger.info("WebSocket 连接已启动")
    
    def stop(self):
        """停止 WebSocket 连接"""
        if not self._running:
            return
        
        self._running = False
        self.subscriptions.clear()
        logger.info("WebSocket 连接已停止")
    
    def subscribe_candles(
        self,
        coin: str,
        interval: str,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        订阅 K 线数据
        
        Args:
            coin: 币种符号，如 "BTC"、"ETH"
            interval: K 线周期，如 "5m"、"1h"
            callback: 收到数据时的回调函数（可选）
        
        Returns:
            是否订阅成功
        """
        if not self._running:
            logger.error("WebSocket 未启动，请先调用 start()")
            return False
        
        if interval not in self.SUPPORTED_INTERVALS:
            logger.error(f"不支持的 K 线周期: {interval}，支持: {self.SUPPORTED_INTERVALS}")
            return False
        
        cache_key = (coin, interval)
        
        # 初始化缓存
        if cache_key not in self.data_cache:
            self.data_cache[cache_key] = deque(maxlen=self.max_cache_size)
        
        # 注册回调
        if callback:
            if cache_key not in self._callbacks:
                self._callbacks[cache_key] = []
            self._callbacks[cache_key].append(callback)
        
        # 订阅
        try:
            self._info.subscribe(
                {
                    "type": "candle",
                    "coin": coin,
                    "interval": interval
                },
                lambda data: self._handle_candle(cache_key, data)
            )
            self.subscriptions.add(cache_key)
            logger.info(f"已订阅 K 线 | {coin} | {interval}")
            return True
        except Exception as e:
            logger.error(f"订阅失败 | {coin} | {interval} | {e}")
            return False
    
    def unsubscribe_candles(self, coin: str, interval: str):
        """
        取消订阅 K 线数据
        
        Args:
            coin: 币种符号
            interval: K 线周期
        """
        cache_key = (coin, interval)
        
        if cache_key in self.subscriptions:
            try:
                self._info.unsubscribe(
                    {
                        "type": "candle",
                        "coin": coin,
                        "interval": interval
                    }
                )
                self.subscriptions.discard(cache_key)
                logger.info(f"已取消订阅 | {coin} | {interval}")
            except Exception as e:
                logger.error(f"取消订阅失败 | {coin} | {interval} | {e}")
    
    def _handle_candle(self, cache_key: tuple[str, str], data: dict):
        """处理接收到的 K 线数据"""
        try:
            # 解析 K 线数据
            candle = self._parse_candle(data)
            if candle is None:
                return
            
            # 更新缓存
            cache = self.data_cache.get(cache_key)
            if cache is not None:
                # 检查是否是更新现有 K 线还是新 K 线
                if cache and cache[-1]["timestamp"] == candle["timestamp"]:
                    # 更新最后一根 K 线
                    cache[-1] = candle
                else:
                    # 添加新 K 线
                    cache.append(candle)
            
            # 触发回调
            callbacks = self._callbacks.get(cache_key, [])
            for cb in callbacks:
                try:
                    cb(candle)
                except Exception as e:
                    logger.error(f"回调执行失败 | {cache_key} | {e}")
        
        except Exception as e:
            logger.error(f"处理 K 线数据失败 | {cache_key} | {e}")
    
    @staticmethod
    def _parse_candle(data: dict) -> Optional[dict]:
        """
        解析 K 线数据
        
        Args:
            data: 原始数据
        
        Returns:
            解析后的 K 线字典，包含 timestamp, open, high, low, close, volume
        """
        try:
            # 根据官方 SDK 的数据格式解析
            # 格式可能是: {"t": timestamp, "o": open, "h": high, "l": low, "c": close, "v": volume}
            # 或者: {"time": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}
            
            if "data" in data:
                candle_data = data["data"]
            else:
                candle_data = data
            
            # 尝试多种格式
            if "t" in candle_data:
                return {
                    "timestamp": int(candle_data["t"]),
                    "open": float(candle_data["o"]),
                    "high": float(candle_data["h"]),
                    "low": float(candle_data["l"]),
                    "close": float(candle_data["c"]),
                    "volume": float(candle_data.get("v", 0))
                }
            elif "time" in candle_data:
                return {
                    "timestamp": int(candle_data["time"]),
                    "open": float(candle_data["open"]),
                    "high": float(candle_data["high"]),
                    "low": float(candle_data["low"]),
                    "close": float(candle_data["close"]),
                    "volume": float(candle_data.get("volume", 0))
                }
            else:
                logger.warning(f"未知的 K 线数据格式: {candle_data}")
                return None
        
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"解析 K 线数据失败: {e} | 数据: {data}")
            return None
    
    def get_cached_data(
        self,
        coin: str,
        interval: str,
        count: Optional[int] = None
    ) -> list[dict]:
        """
        获取缓存的 K 线数据
        
        Args:
            coin: 币种符号
            interval: K 线周期
            count: 返回的数量（从最新开始），None 表示全部
        
        Returns:
            K 线数据列表
        """
        cache_key = (coin, interval)
        cache = self.data_cache.get(cache_key)
        
        if cache is None:
            return []
        
        data = list(cache)
        if count is not None:
            data = data[-count:]
        
        return data
    
    def get_cached_dataframe(
        self,
        coin: str,
        interval: str,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """
        获取缓存的 K 线数据（DataFrame 格式）
        
        Args:
            coin: 币种符号
            interval: K 线周期
            count: 返回的数量
        
        Returns:
            DataFrame，索引为 Timestamp
        """
        data = self.get_cached_data(coin, interval, count)
        
        if not data:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        df = pd.DataFrame(data)
        df["Timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        df = df.set_index("Timestamp")[["Open", "High", "Low", "Close", "Volume"]]
        df = df.sort_index()
        
        return df
    
    def has_enough_data(self, coin: str, interval: str, required_bars: int) -> bool:
        """
        检查缓存中是否有足够的数据
        
        Args:
            coin: 币种符号
            interval: K 线周期
            required_bars: 需要的 K 线数量
        
        Returns:
            是否有足够数据
        """
        cache_key = (coin, interval)
        cache = self.data_cache.get(cache_key)
        
        if cache is None:
            return False
        
        return len(cache) >= required_bars
    
    def get_subscription_status(self) -> dict:
        """获取订阅状态"""
        return {
            "running": self._running,
            "subscriptions": list(self.subscriptions),
            "cache_sizes": {
                f"{coin}_{interval}": len(cache)
                for (coin, interval), cache in self.data_cache.items()
            }
        }
    
    @property
    def is_running(self) -> bool:
        """WebSocket 是否正在运行"""
        return self._running

