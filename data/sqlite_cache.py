"""
SQLite 缓存模块

用于存储历史 K 线数据，避免重复下载。
支持增量更新，只下载缺失的数据。
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SQLiteCache:
    """SQLite K 线数据缓存"""
    
    def __init__(self, db_path: str = "hyperliquid_data.db"):
        """
        初始化 SQLite 缓存
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            
            # 创建索引以加速查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe 
                ON ohlcv(symbol, timeframe)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON ohlcv(symbol, timeframe, timestamp)
            """)
            
            conn.commit()
            logger.debug(f"数据库初始化完成 | 路径: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """
        保存 OHLCV 数据到缓存
        
        Args:
            symbol: 交易对，如 "BTC/USDC:USDC"
            timeframe: K 线周期，如 "5m"
            df: 包含 OHLCV 数据的 DataFrame（索引为 Timestamp）
        
        Returns:
            插入的行数
        """
        if df.empty:
            return 0
        
        # 准备数据
        records = []
        for timestamp, row in df.iterrows():
            # 将 Timestamp 转换为毫秒时间戳
            ts_ms = int(timestamp.timestamp() * 1000)
            records.append((
                symbol,
                timeframe,
                ts_ms,
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                float(row['Volume'])
            ))
        
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO ohlcv 
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
        
        logger.debug(f"缓存数据保存 | {symbol} | {timeframe} | {len(records)} 条")
        return len(records)
    
    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        从缓存获取 OHLCV 数据
        
        Args:
            symbol: 交易对
            timeframe: K 线周期
            since_ms: 起始时间戳（毫秒）
            until_ms: 结束时间戳（毫秒）
            limit: 返回的最大行数
        
        Returns:
            DataFrame，索引为 Timestamp
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]
        
        if since_ms is not None:
            query += " AND timestamp >= ?"
            params.append(since_ms)
        
        if until_ms is not None:
            query += " AND timestamp <= ?"
            params.append(until_ms)
        
        query += " ORDER BY timestamp ASC"
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        # 转换为 DataFrame
        df = pd.DataFrame(rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.set_index("Timestamp").sort_index()
        
        logger.debug(f"缓存数据读取 | {symbol} | {timeframe} | {len(df)} 条")
        return df
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        获取缓存中最新的时间戳
        
        Args:
            symbol: 交易对
            timeframe: K 线周期
        
        Returns:
            最新时间戳（毫秒），如果没有数据返回 None
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT MAX(timestamp) FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            result = cursor.fetchone()
        
        return result[0] if result and result[0] is not None else None
    
    def get_oldest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        获取缓存中最早的时间戳
        
        Args:
            symbol: 交易对
            timeframe: K 线周期
        
        Returns:
            最早时间戳（毫秒），如果没有数据返回 None
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT MIN(timestamp) FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            result = cursor.fetchone()
        
        return result[0] if result and result[0] is not None else None
    
    def get_data_count(self, symbol: str, timeframe: str) -> int:
        """
        获取缓存中的数据条数
        
        Args:
            symbol: 交易对
            timeframe: K 线周期
        
        Returns:
            数据条数
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            result = cursor.fetchone()
        
        return result[0] if result else 0
    
    def clear_symbol(self, symbol: str, timeframe: Optional[str] = None):
        """
        清除指定交易对的缓存数据
        
        Args:
            symbol: 交易对
            timeframe: K 线周期（可选，不指定则清除所有周期）
        """
        with self._get_connection() as conn:
            if timeframe:
                conn.execute("""
                    DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
            else:
                conn.execute("""
                    DELETE FROM ohlcv WHERE symbol = ?
                """, (symbol,))
            conn.commit()
        
        logger.info(f"缓存已清除 | {symbol} | {timeframe or '所有周期'}")
    
    def get_all_symbols(self) -> list[str]:
        """获取缓存中所有的交易对"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT symbol FROM ohlcv
            """)
            return [row[0] for row in cursor.fetchall()]
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT symbol, timeframe, COUNT(*) as count,
                       MIN(timestamp) as oldest,
                       MAX(timestamp) as newest
                FROM ohlcv
                GROUP BY symbol, timeframe
            """)
            rows = cursor.fetchall()
        
        stats = {}
        for symbol, timeframe, count, oldest, newest in rows:
            if symbol not in stats:
                stats[symbol] = {}
            stats[symbol][timeframe] = {
                "count": count,
                "oldest_ms": oldest,
                "newest_ms": newest
            }
        
        return stats

