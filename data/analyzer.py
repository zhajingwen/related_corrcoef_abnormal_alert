"""
相关系数分析器模块

分析山寨币与 BTC 的皮尔逊相关系数，识别存在时间差套利空间的异常币种。
基于 DataManager 获取数据，支持 WebSocket 实时数据和 REST 历史数据。
"""

import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import numpy as np
import pandas as pd

from .manager import DataManager

# 尝试导入飞书通知（可选）
try:
    from utils.lark_bot import sender
    from utils.config import lark_bot_id
    HAS_LARK_BOT = True
except ImportError:
    HAS_LARK_BOT = False
    lark_bot_id = None


def setup_logging(log_file: str = "analyzer.log", level: int = logging.INFO) -> logging.Logger:
    """
    配置日志系统，支持控制台和文件输出
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        配置好的 logger 实例
    """
    log = logging.getLogger("data.analyzer")
    
    # 避免重复添加 handlers
    if log.handlers:
        return log
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 文件处理器（10MB轮转，保留5个备份）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # 配置 logger
    log.setLevel(level)
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    
    return log


logger = setup_logging()


class DelayCorrelationAnalyzer:
    """
    山寨币与 BTC 相关系数分析器
    
    识别短期低相关但长期高相关的异常币种，这类币种存在时间差套利机会。
    """
    
    # 相关系数计算所需的最小数据点数
    MIN_POINTS_FOR_CORR_CALC = 10
    # 数据分析所需的最小数据点数
    MIN_DATA_POINTS_FOR_ANALYSIS = 50
    
    # 异常模式检测阈值
    LONG_TERM_CORR_THRESHOLD = 0.6   # 长期相关系数阈值
    SHORT_TERM_CORR_THRESHOLD = 0.3  # 短期相关系数阈值
    CORR_DIFF_THRESHOLD = 0.5        # 相关系数差值阈值
    
    def __init__(
        self,
        exchange_name: str = "hyperliquid",
        db_path: str = "hyperliquid_data.db",
        use_websocket: bool = True,
        testnet: bool = False,
        default_timeframes: Optional[list[str]] = None,
        default_periods: Optional[list[str]] = None
    ):
        """
        初始化分析器
        
        Args:
            exchange_name: 交易所名称
            db_path: SQLite 数据库路径
            use_websocket: 是否启用 WebSocket
            testnet: 是否使用测试网
            default_timeframes: K 线颗粒度列表
            default_periods: 数据周期列表
        """
        self.exchange_name = exchange_name
        self.timeframes = default_timeframes or ["1m", "5m"]
        self.periods = default_periods or ["1d", "7d", "30d", "60d"]
        self.btc_symbol = "BTC/USDC:USDC"
        
        # 初始化数据管理器
        self.data_manager = DataManager(
            exchange_name=exchange_name,
            db_path=db_path,
            use_websocket=use_websocket,
            testnet=testnet
        )
        
        # 飞书通知配置
        if HAS_LARK_BOT and lark_bot_id:
            self.lark_hook = f'https://open.feishu.cn/open-apis/bot/v2/hook/{lark_bot_id}'
        else:
            self.lark_hook = None
            if not HAS_LARK_BOT:
                logger.warning("飞书通知模块未找到，通知功能不可用")
            elif not lark_bot_id:
                logger.warning("环境变量 LARKBOT_ID 未设置，飞书通知功能不可用")
        
        logger.info(
            f"分析器初始化 | 交易所: {exchange_name} | "
            f"时间周期: {self.timeframes} | 数据周期: {self.periods}"
        )
    
    def initialize(self):
        """初始化：启动数据管理器"""
        self.data_manager.initialize()
        
        # 预取 BTC 数据
        logger.info("预取 BTC 历史数据...")
        self.data_manager.prefetch_btc_data(self.timeframes, self.periods)
    
    def shutdown(self):
        """关闭：停止数据管理器"""
        self.data_manager.shutdown()
    
    @staticmethod
    def find_optimal_delay(btc_ret: np.ndarray, alt_ret: np.ndarray, max_lag: int = 48) -> tuple:
        """
        寻找最优延迟 τ*
        
        通过计算不同延迟下 BTC 和山寨币收益率的相关系数，找出使相关系数最大的延迟值。
        tau_star > 0 表示山寨币滞后于 BTC，存在时间差套利机会。
        
        Args:
            btc_ret: BTC 收益率数组
            alt_ret: 山寨币收益率数组
            max_lag: 最大延迟值
        
        Returns:
            (tau_star, corrs, max_corr): 最优延迟、相关系数列表、最大相关系数
        """
        corrs = []
        lags = list(range(0, max_lag + 1))
        arr_len = len(btc_ret)
        
        for lag in lags:
            # 检查 lag 是否超过数组长度
            if lag > 0 and lag >= arr_len:
                corrs.append(np.nan)
                continue
            
            if lag > 0:
                # ALT 滞后 BTC: 比较 BTC[t] 与 ALT[t+lag]
                x = btc_ret[:-lag]
                y = alt_ret[lag:]
            else:
                x = btc_ret
                y = alt_ret
            
            m = min(len(x), len(y))
            
            if m < DelayCorrelationAnalyzer.MIN_POINTS_FOR_CORR_CALC:
                corrs.append(np.nan)
                continue
            
            corr = np.corrcoef(x[:m], y[:m])[0, 1]
            corrs.append(np.nan if np.isnan(corr) else corr)
        
        # 找出最大相关系数对应的延迟值
        valid_corrs = np.array(corrs)
        valid_mask = ~np.isnan(valid_corrs)
        
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(valid_corrs[valid_mask])]
            tau_star = lags[best_idx]
            max_corr = valid_corrs[best_idx]
        else:
            tau_star = 0
            max_corr = np.nan
        
        return tau_star, corrs, max_corr
    
    def _get_btc_data(self, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        """获取 BTC 数据"""
        return self.data_manager.get_btc_data(timeframe, period)
    
    def _get_coin_data(self, symbol: str, timeframe: str, period: str) -> Optional[pd.DataFrame]:
        """获取币种数据"""
        try:
            df = self.data_manager.get_ohlcv(symbol, timeframe, period)
            return df if not df.empty else None
        except Exception as e:
            logger.warning(f"获取数据失败 | {symbol} | {timeframe}/{period} | {e}")
            return None
    
    def _align_and_validate_data(
        self,
        btc_df: pd.DataFrame,
        alt_df: pd.DataFrame,
        coin: str,
        timeframe: str,
        period: str
    ) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        对齐和验证 BTC 与山寨币数据
        
        Args:
            btc_df: BTC 数据 DataFrame
            alt_df: 山寨币数据 DataFrame
            coin: 币种名称（用于日志）
            timeframe: 时间周期
            period: 数据周期
        
        Returns:
            成功返回对齐后的 (btc_df, alt_df)，失败返回 None
        """
        # 对齐时间索引
        common_idx = btc_df.index.intersection(alt_df.index)
        btc_df_aligned = btc_df.loc[common_idx]
        alt_df_aligned = alt_df.loc[common_idx]
        
        # 数据验证：检查数据量
        if len(btc_df_aligned) < self.MIN_DATA_POINTS_FOR_ANALYSIS:
            logger.warning(f"数据量不足，跳过 | 币种: {coin} | {timeframe}/{period}")
            return None
        
        if len(alt_df_aligned) < self.MIN_DATA_POINTS_FOR_ANALYSIS:
            logger.warning(f"数据量不足，跳过 | 币种: {coin} | {timeframe}/{period}")
            return None
        
        return btc_df_aligned, alt_df_aligned
    
    def _analyze_single_combination(
        self,
        coin: str,
        timeframe: str,
        period: str
    ) -> Optional[tuple]:
        """
        分析单个 timeframe/period 组合
        
        Returns:
            成功返回 (correlation, timeframe, period, tau_star)，失败返回 None
        """
        logger.debug(f"下载数据 | 币种: {coin} | {timeframe}/{period}")
        
        btc_df = self._get_btc_data(timeframe, period)
        if btc_df is None:
            return None
        
        alt_df = self._get_coin_data(coin, timeframe, period)
        if alt_df is None:
            return None
        
        # 对齐和验证数据
        aligned_data = self._align_and_validate_data(btc_df, alt_df, coin, timeframe, period)
        if aligned_data is None:
            return None
        
        btc_df_aligned, alt_df_aligned = aligned_data
        
        tau_star, _, corr = self.find_optimal_delay(
            btc_df_aligned['return'].values,
            alt_df_aligned['return'].values
        )
        
        logger.debug(
            f"分析结果 | timeframe: {timeframe} | period: {period} | "
            f"tau_star: {tau_star} | 相关系数: {corr:.4f}"
        )
        
        return (corr, timeframe, period, tau_star)
    
    def _detect_anomaly_pattern(self, results: list) -> tuple[bool, float]:
        """
        检测异常模式：短期低相关但长期高相关
        
        Returns:
            (is_anomaly, diff_amount): 是否异常模式、相关系数差值
        """
        short_periods = ['1d']
        long_periods = ['7d', '30d', '60d']
        
        short_term_corrs = [x[0] for x in results if x[2] in short_periods]
        long_term_corrs = [x[0] for x in results if x[2] in long_periods]
        
        if not short_term_corrs or not long_term_corrs:
            return False, 0
        
        min_short_corr = min(short_term_corrs)
        max_long_corr = max(long_term_corrs)
        
        logger.debug(f"相关系数检测 | 短期最小: {min_short_corr:.4f} | 长期最大: {max_long_corr:.4f}")
        
        if max_long_corr > self.LONG_TERM_CORR_THRESHOLD and min_short_corr < self.SHORT_TERM_CORR_THRESHOLD:
            diff_amount = max_long_corr - min_short_corr
            if diff_amount > self.CORR_DIFF_THRESHOLD:
                return True, diff_amount
            # 短期存在明显滞后时也触发
            if any(tau_star > 0 for _, _, period, tau_star in results if period == '1d'):
                return True, diff_amount
        
        return False, 0
    
    def _output_results(self, coin: str, results: list, diff_amount: float):
        """输出异常模式的分析结果"""
        df_results = pd.DataFrame([
            {'相关系数': corr, '时间周期': tf, '数据周期': p, '最优延迟': ts}
            for corr, tf, p, ts in results
        ])
        
        logger.info(
            f"发现异常币种 | 交易所: {self.exchange_name} | 币种: {coin} | 差值: {diff_amount:.2f}"
        )
        
        # 飞书消息内容
        content = f"{self.exchange_name}\n\n{coin} 相关系数分析结果\n{df_results.to_string(index=False)}\n"
        content += f"\n差值: {diff_amount:.2f}"
        logger.debug(f"详细分析结果:\n{df_results.to_string(index=False)}")
        
        # 发送飞书通知
        if self.lark_hook and HAS_LARK_BOT:
            try:
                sender(content, self.lark_hook)
            except Exception as e:
                logger.error(f"飞书通知发送失败 | {e}")
        else:
            logger.warning(f"飞书通知未发送（未配置）| 币种: {coin}")
    
    def one_coin_analysis(self, coin: str) -> bool:
        """
        分析单个币种与 BTC 的相关系数，识别异常模式
        
        Args:
            coin: 币种交易对名称，如 "ETH/USDC:USDC"
        
        Returns:
            是否发现异常模式
        """
        results = []
        
        for timeframe in self.timeframes:
            for period in self.periods:
                try:
                    result = self._analyze_single_combination(coin, timeframe, period)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"处理失败 | {coin} | {timeframe}/{period} | {e}")
        
        # 过滤 NaN 并按相关系数降序排序
        valid_results = [(corr, tf, p, ts) for corr, tf, p, ts in results if not np.isnan(corr)]
        valid_results = sorted(valid_results, key=lambda x: x[0], reverse=True)
        
        if not valid_results:
            logger.warning(f"数据不足，无法分析 | 币种: {coin}")
            return False
        
        is_anomaly, diff_amount = self._detect_anomaly_pattern(valid_results)
        
        if is_anomaly:
            self._output_results(coin, valid_results, diff_amount)
            return True
        else:
            logger.debug(f"常规数据 | 币种: {coin}")
            return False
    
    def run(self):
        """分析交易所中所有 USDC 永续合约交易对"""
        logger.info(
            f"启动分析器 | 交易所: {self.exchange_name} | "
            f"时间周期: {self.timeframes} | 数据周期: {self.periods}"
        )
        
        # 初始化
        self.initialize()
        
        try:
            # 获取所有交易对
            usdc_coins = self.data_manager.get_usdc_perpetuals()
            total = len(usdc_coins)
            anomaly_count = 0
            skip_count = 0
            start_time = time.time()
            
            logger.info(f"发现 {total} 个 USDC 永续合约交易对")
            
            # 进度里程碑
            milestones = {max(1, int(total * p)) for p in [0.25, 0.5, 0.75, 1.0]}
            
            for idx, coin in enumerate(usdc_coins, 1):
                logger.debug(f"检查币种: {coin}")
                
                try:
                    result = self.one_coin_analysis(coin)
                    if result:
                        anomaly_count += 1
                except Exception as e:
                    logger.error(f"分析币种失败 | {coin} | {e}")
                    skip_count += 1
                
                # 在里程碑位置打印进度
                if idx in milestones:
                    logger.info(f"分析进度: {idx}/{total} ({idx * 100 // total}%)")
                
                time.sleep(0.5)  # 降低请求频率
            
            elapsed = time.time() - start_time
            logger.info(
                f"分析完成 | 交易所: {self.exchange_name} | "
                f"总数: {total} | 异常: {anomaly_count} | 跳过: {skip_count} | "
                f"耗时: {elapsed:.1f}s | 平均: {elapsed/max(total, 1):.2f}s/币种"
            )
        
        finally:
            self.shutdown()
    
    def run_single(self, coin: str):
        """分析单个币种（用于测试）"""
        self.initialize()
        try:
            self.one_coin_analysis(coin)
        finally:
            self.shutdown()

