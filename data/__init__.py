"""
Data 模块

提供 WebSocket + REST 混合数据架构，支持：
- SQLite 本地缓存
- REST API 历史数据下载（带速率限制）
- WebSocket 实时数据订阅
- 统一的数据管理接口
- 相关系数分析器

使用示例:
    from data import DataManager, DelayCorrelationAnalyzer
    
    # 使用数据管理器获取数据
    manager = DataManager()
    manager.initialize()
    df = manager.get_ohlcv("BTC/USDC:USDC", "5m", "7d")
    manager.shutdown()
    
    # 运行分析器
    analyzer = DelayCorrelationAnalyzer()
    analyzer.run()

注意：如果当前目录存在 hyperliquid.py 文件，会遮蔽 hyperliquid SDK。
解决方案：
1. 将 hyperliquid.py 重命名为其他名称（如 hyperliquid_analyzer.py）
2. 或使用 --no-websocket 参数禁用 WebSocket 功能
"""

import sys
import os

# 修复 hyperliquid.py 遮蔽 SDK 的问题
# 确保 site-packages 的优先级高于当前工作目录
def _fix_import_path():
    """将 site-packages 路径提升到最前面，避免本地文件遮蔽安装的包"""
    cwd = os.getcwd()
    site_packages = []
    other_paths = []
    
    for path in sys.path:
        if 'site-packages' in path:
            site_packages.append(path)
        elif path != cwd and path != '':
            other_paths.append(path)
    
    # 重新排序：site-packages 优先
    sys.path = site_packages + other_paths + [cwd, '']

_fix_import_path()

from .sqlite_cache import SQLiteCache
from .rest_client import RESTClient
from .websocket_client import WebSocketClient, HAS_HYPERLIQUID_SDK
from .manager import DataManager
from .analyzer import DelayCorrelationAnalyzer

__all__ = [
    "SQLiteCache",
    "RESTClient",
    "WebSocketClient",
    "DataManager",
    "DelayCorrelationAnalyzer",
    "HAS_HYPERLIQUID_SDK",
]

__version__ = "1.0.0"

