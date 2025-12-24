# 功能：分析山寨币与BTC的皮尔逊相关系数，识别存在时间差套利空间的异常币种
# 原理：通过计算不同时间周期和延迟下的相关系数，找出短期低相关但长期高相关的币种

import ccxt
import time
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from retry import retry
from utils.lark_bot import sender
from utils.config import lark_bot_id


def setup_logging(log_file="hyperliquid.log", level=logging.DEBUG):
    """
    配置日志系统，支持控制台和文件输出
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        配置好的 logger 实例
    """
    log = logging.getLogger(__name__)
    
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
    log.propagate = False  # 阻止日志传播到根 logger，避免重复打印
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    
    return log


logger = setup_logging()


class DelayCorrelationAnalyzer:
    """
    山寨币与BTC相关系数分析器
    
    识别短期低相关但长期高相关的异常币种，这类币种存在时间差套利机会。
    """
    # 相关系数计算所需的最小数据点数
    MIN_POINTS_FOR_CORR_CALC = 10
    # 数据分析所需的最小数据点数
    MIN_DATA_POINTS_FOR_ANALYSIS = 50
    
    # 异常模式检测阈值
    LONG_TERM_CORR_THRESHOLD = 0.6  # 长期相关系数阈值
    SHORT_TERM_CORR_THRESHOLD = 0.3  # 短期相关系数阈值
    CORR_DIFF_THRESHOLD = 0.5  # 相关系数差值阈值
    
    def __init__(self, exchange_name="kucoin", timeout=30000, default_timeframes=None, default_periods=None):
        """
        初始化分析器
        
        Args:
            exchange_name: 交易所名称，支持ccxt库支持的所有交易所
            timeout: 请求超时时间（毫秒）
            default_timeframes: K线颗粒度列表，如 ["1m", "5m"]
            default_periods: 数据周期列表，如 ["1d", "7d", "30d"]
        """
        self.exchange_name = exchange_name
        self.exchange = getattr(ccxt, exchange_name)({
            "timeout": timeout,
            "enableRateLimit": True,
            "rateLimit": 1500
        })
        self.timeframes = default_timeframes or ["1m", "5m"]
        self.periods = default_periods or ["1d", "7d", "30d"]
        self.btc_symbol = "BTC/USDC:USDC"
        self.btc_df_cache = {}
        self.alt_df_cache = {}  # 山寨币数据缓存
        
        # 检查 lark_bot_id 是否有效
        if not lark_bot_id:
            logger.warning("环境变量 LARKBOT_ID 未设置，飞书通知功能将不可用")
            self.lark_hook = None
        else:
            self.lark_hook = f'https://open.feishu.cn/open-apis/bot/v2/hook/{lark_bot_id}'

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        """
        将 timeframe 字符串转换为分钟数
        
        支持的格式：
        - 分钟：1m, 5m, 15m, 30m
        - 小时：1h, 4h, 12h
        - 天：1d, 3d
        - 周：1w
        
        Args:
            timeframe: K线时间周期字符串
        
        Returns:
            对应的分钟数
        
        Raises:
            ValueError: 不支持的 timeframe 格式
        """
        unit_multipliers = {
            'm': 1,
            'h': 60,
            'd': 24 * 60,
            'w': 7 * 24 * 60,
        }
        
        unit = timeframe[-1].lower()
        if unit not in unit_multipliers:
            raise ValueError(f"不支持的 timeframe 格式: {timeframe}，支持的单位: m, h, d, w")
        
        try:
            value = int(timeframe[:-1])
        except ValueError:
            raise ValueError(f"无效的 timeframe 格式: {timeframe}，数值部分必须是整数")
        
        return value * unit_multipliers[unit]
    
    @staticmethod
    def _period_to_bars(period: str, timeframe: str) -> int:
        """将时间周期转换为K线总条数"""
        days = int(period.rstrip('d'))
        timeframe_minutes = DelayCorrelationAnalyzer._timeframe_to_minutes(timeframe)
        bars_per_day = int(24 * 60 / timeframe_minutes)
        return days * bars_per_day
    
    def _safe_download(self, symbol: str, period: str, timeframe: str, coin: str = None) -> pd.DataFrame | None:
        """
        安全下载数据，失败时返回None并记录日志
        
        Args:
            symbol: 交易对名称
            period: 数据周期
            timeframe: K线时间周期
            coin: 用于日志的币种名称（可选）
        
        Returns:
            成功返回DataFrame，失败返回None
        """
        display_name = coin or symbol
        return self._safe_execute(
            self.download_ccxt_data,
            symbol, period=period, timeframe=timeframe,
            error_msg=f"下载 {display_name} 的 {timeframe}/{period} 数据失败"
        )
    
    @retry(tries=10, delay=5, backoff=2, logger=logger)
    def download_ccxt_data(self, symbol: str, period: str, timeframe: str) -> pd.DataFrame:
        """
        从交易所下载OHLCV历史数据
        
        Args:
            symbol: 交易对名称，如 "BTC/USDC"
            period: 数据周期，如 "30d"
            timeframe: K线时间周期，如 "5m"
        
        Returns:
            包含 Open/High/Low/Close/Volume/return/volume_usd 列的DataFrame
        """
        target_bars = self._period_to_bars(period, timeframe)
        ms_per_bar = self._timeframe_to_minutes(timeframe) * 60 * 1000
        now_ms = self.exchange.milliseconds()
        since = now_ms - target_bars * ms_per_bar

        all_rows = []
        fetched = 0
        
        while True:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
            if not ohlcv:
                break
            
            all_rows.extend(ohlcv)
            fetched += len(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(ohlcv) < 1500 or fetched >= target_bars:
                break
            
            # 请求间隔：添加 1.5 秒延迟，确保即使 ccxt 内部发起多次请求也有足够间隔
            # 对 Hyperliquid 来说，1.5 秒是安全的间隔
            time.sleep(1.5)

        if not all_rows:
            logger.debug(f"交易对无历史数据（API返回空列表）| 币种: {symbol} | {timeframe}/{period}")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "return", "volume_usd"])

        df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.set_index("Timestamp").sort_index()
        df['return'] = df['Close'].pct_change().fillna(0)
        df['volume_usd'] = df['Volume'] * df['Close']
        
        return df
    
    @staticmethod
    def find_optimal_delay(btc_ret, alt_ret, max_lag=48):
        """
        寻找最优延迟 τ*
        
        通过计算不同延迟下BTC和山寨币收益率的相关系数，找出使相关系数最大的延迟值。
        tau_star > 0 表示山寨币滞后于BTC，存在时间差套利机会。
        
        Args:
            btc_ret: BTC收益率数组
            alt_ret: 山寨币收益率数组
            max_lag: 最大延迟值
        
        Returns:
            (tau_star, corrs, max_related_matrix): 最优延迟、相关系数列表、最大相关系数
        """
        corrs = []
        lags = list(range(0, max_lag + 1))
        arr_len = len(btc_ret)
        
        for lag in lags:
            # 检查 lag 是否超过数组长度，避免空数组切片
            if lag > 0 and lag >= arr_len:
                corrs.append(np.nan)
                continue
            
            if lag > 0:
                # ALT滞后BTC: 比较 BTC[t] 与 ALT[t+lag]
                x = btc_ret[:-lag]
                y = alt_ret[lag:]
            else:
                x = btc_ret
                y = alt_ret
            
            m = min(len(x), len(y))
            
            if m < DelayCorrelationAnalyzer.MIN_POINTS_FOR_CORR_CALC:
                corrs.append(np.nan)
                continue
            
            related_matrix = np.corrcoef(x[:m], y[:m])[0, 1]
            corrs.append(np.nan if np.isnan(related_matrix) else related_matrix)
        
        # 找出最大相关系数对应的延迟值
        valid_corrs = np.array(corrs)
        valid_mask = ~np.isnan(valid_corrs)
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[np.argmax(valid_corrs[valid_mask])]
            tau_star = lags[best_idx]
            max_related_matrix = valid_corrs[best_idx]
        else:
            tau_star = 0
            max_related_matrix = np.nan
        
        return tau_star, corrs, max_related_matrix
    
    def _get_btc_data(self, timeframe: str, period: str) -> pd.DataFrame | None:
        """获取BTC数据（带缓存和智能切片）"""
        cache_key = (timeframe, period)
        if cache_key in self.btc_df_cache:
            logger.debug(f"BTC数据缓存命中 | {timeframe}/{period}")
            return self.btc_df_cache[cache_key].copy()
        
        # 对于1m/1d、1m/7d、5m/1d和5m/7d，尝试从对应timeframe的30d数据中切片
        if timeframe in ["1m", "5m"] and period in ["1d", "7d"]:
            source_cache_key = (timeframe, "30d")
            if source_cache_key in self.btc_df_cache:
                logger.debug(f"BTC数据从30d切片生成 | {timeframe}/{period}")
                source_df = self.btc_df_cache[source_cache_key]
                target_bars = self._period_to_bars(period, timeframe)
                sliced_df = source_df.tail(target_bars).copy()
                # 重新计算return列
                sliced_df['return'] = sliced_df['Close'].pct_change().fillna(0)
                sliced_df['volume_usd'] = sliced_df['Volume'] * sliced_df['Close']
                # 缓存切片后的数据
                self.btc_df_cache[cache_key] = sliced_df
                return sliced_df
            else:
                # 如果没有30d数据，先下载30d数据并缓存
                logger.debug(f"BTC数据首次下载并缓存30d数据，用于后续切片生成1d/7d | {timeframe}/{period}")
                btc_30d_df = self._safe_download(self.btc_symbol, "30d", timeframe)
                if btc_30d_df is None:
                    return None
                self.btc_df_cache[source_cache_key] = btc_30d_df
                # 从30d数据中切片
                target_bars = self._period_to_bars(period, timeframe)
                sliced_df = btc_30d_df.tail(target_bars).copy()
                sliced_df['return'] = sliced_df['Close'].pct_change().fillna(0)
                sliced_df['volume_usd'] = sliced_df['Volume'] * sliced_df['Close']
                self.btc_df_cache[cache_key] = sliced_df
                return sliced_df
        
        logger.debug(f"BTC数据缓存未命中，开始下载 | {timeframe}/{period}")
        btc_df = self._safe_download(self.btc_symbol, period, timeframe)
        if btc_df is None:
            return None
        self.btc_df_cache[cache_key] = btc_df
        return btc_df.copy()
    
    def _get_alt_data(self, symbol: str, period: str, timeframe: str, coin: str = None) -> pd.DataFrame | None:
        """
        获取山寨币数据（带缓存和智能切片）
        
        对于1m/1d、1m/7d、5m/1d和5m/7d，从对应timeframe的30d数据中本地切片生成，减少API请求
        
        Args:
            symbol: 交易对名称
            period: 数据周期
            timeframe: K线时间周期
            coin: 用于日志的币种名称（可选）
        
        Returns:
            成功返回DataFrame，失败返回None
        """
        display_name = coin or symbol
        cache_key = (symbol, timeframe, period)
        
        # 检查缓存
        if cache_key in self.alt_df_cache:
            cached_df = self.alt_df_cache[cache_key]
            # 验证缓存的数据是否为空
            if cached_df.empty or len(cached_df) == 0:
                logger.warning(f"山寨币数据缓存命中但数据为空，跳过 | 币种: {display_name} | {timeframe}/{period}")
                return None
            logger.debug(f"山寨币数据缓存命中 | 币种: {display_name} | {timeframe}/{period}")
            return cached_df.copy()
        
        # 对于1m/1d、1m/7d、5m/1d和5m/7d，尝试从对应timeframe的30d数据中切片
        if timeframe in ["1m", "5m"] and period in ["1d", "7d"]:
            source_cache_key = (symbol, timeframe, "30d")
            if source_cache_key in self.alt_df_cache:
                source_df = self.alt_df_cache[source_cache_key]
                # 检查源数据是否为空
                if source_df.empty or len(source_df) == 0:
                    logger.warning(f"山寨币30d源数据为空，无法切片 | 币种: {display_name} | {timeframe}/{period}")
                    return None
                logger.debug(f"山寨币数据从30d切片生成 | 币种: {display_name} | {timeframe}/{period}")
                target_bars = self._period_to_bars(period, timeframe)
                sliced_df = source_df.tail(target_bars).copy()
                # 重新计算return列
                sliced_df['return'] = sliced_df['Close'].pct_change().fillna(0)
                sliced_df['volume_usd'] = sliced_df['Volume'] * sliced_df['Close']
                # 验证切片后的数据量
                if len(sliced_df) < self.MIN_DATA_POINTS_FOR_ANALYSIS:
                    logger.warning(f"山寨币切片后数据量不足，不缓存 | 币种: {display_name} | {timeframe}/{period} | 数据量: {len(sliced_df)}")
                    return None
                # 缓存切片后的数据
                self.alt_df_cache[cache_key] = sliced_df
                return sliced_df
            else:
                # 如果没有30d数据，先下载30d数据并缓存
                logger.debug(f"山寨币数据首次下载并缓存30d数据，用于后续切片生成1d/7d | 币种: {display_name} | {timeframe}/{period}")
                alt_30d_df = self._safe_download(symbol, "30d", timeframe, coin)
                if alt_30d_df is None:
                    return None
                # 验证下载的数据是否为空
                if alt_30d_df.empty or len(alt_30d_df) == 0:
                    logger.warning(f"山寨币30d数据不存在（空数据），不缓存 | 币种: {display_name} | {timeframe}/30d")
                    return None
                # 验证数据量是否足够
                if len(alt_30d_df) < self.MIN_DATA_POINTS_FOR_ANALYSIS:
                    logger.warning(f"山寨币30d数据量不足，不缓存 | 币种: {display_name} | {timeframe}/30d | 数据量: {len(alt_30d_df)}")
                    return None
                self.alt_df_cache[source_cache_key] = alt_30d_df
                # 从30d数据中切片
                target_bars = self._period_to_bars(period, timeframe)
                sliced_df = alt_30d_df.tail(target_bars).copy()
                sliced_df['return'] = sliced_df['Close'].pct_change().fillna(0)
                sliced_df['volume_usd'] = sliced_df['Volume'] * sliced_df['Close']
                # 验证切片后的数据量
                if len(sliced_df) < self.MIN_DATA_POINTS_FOR_ANALYSIS:
                    logger.warning(f"山寨币切片后数据量不足，不缓存 | 币种: {display_name} | {timeframe}/{period} | 数据量: {len(sliced_df)}")
                    return None
                self.alt_df_cache[cache_key] = sliced_df
                return sliced_df
        
        # 对于其他组合，直接下载并缓存
        logger.debug(f"山寨币数据缓存未命中，开始下载 | 币种: {display_name} | {timeframe}/{period}")
        alt_df = self._safe_download(symbol, period, timeframe, coin)
        if alt_df is None:
            return None
        # 验证下载的数据是否为空
        if alt_df.empty or len(alt_df) == 0:
            logger.warning(f"山寨币数据不存在（空数据），不缓存 | 币种: {display_name} | {timeframe}/{period}")
            return None
        # 验证数据量是否足够
        if len(alt_df) < self.MIN_DATA_POINTS_FOR_ANALYSIS:
            logger.warning(f"山寨币数据量不足，不缓存 | 币种: {display_name} | {timeframe}/{period} | 数据量: {len(alt_df)}")
            return None
        self.alt_df_cache[cache_key] = alt_df
        return alt_df.copy()
    
    @staticmethod
    def _safe_execute(func, *args, error_msg: str = None, log_error: bool = True, **kwargs):
        """
        安全执行函数，统一错误处理
        
        Args:
            func: 要执行的函数
            *args: 函数的位置参数
            error_msg: 自定义错误消息（可选）
            log_error: 是否记录错误日志（默认True）
            **kwargs: 函数的关键字参数
        
        Returns:
            函数返回值，如果发生异常返回 None
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_error and error_msg:
                logger.warning(f"{error_msg} | {type(e).__name__}: {str(e)}")
            return None
    
    def _align_and_validate_data(self, btc_df: pd.DataFrame, alt_df: pd.DataFrame, 
                                  coin: str, timeframe: str, period: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """
        对齐和验证BTC与山寨币数据
        
        Args:
            btc_df: BTC数据DataFrame
            alt_df: 山寨币数据DataFrame
            coin: 币种名称（用于日志）
            timeframe: 时间周期
            period: 数据周期
        
        Returns:
            成功返回对齐后的 (btc_df, alt_df)，失败返回 None
        """
        # 检查数据是否存在（区分"数据不存在"和"数据量不足"）
        if alt_df.empty or len(alt_df) == 0:
            logger.warning(f"数据不存在（空数据），跳过 | 币种: {coin} | {timeframe}/{period}")
            return None
        
        # 对齐时间索引
        common_idx = btc_df.index.intersection(alt_df.index)
        btc_df_aligned = btc_df.loc[common_idx]
        alt_df_aligned = alt_df.loc[common_idx]
        
        # 数据验证：检查数据量（数据存在但不足）
        if len(btc_df_aligned) < self.MIN_DATA_POINTS_FOR_ANALYSIS or len(alt_df_aligned) < self.MIN_DATA_POINTS_FOR_ANALYSIS:
            logger.warning(f"数据量不足，跳过 | 币种: {coin} | {timeframe}/{period} | BTC数据量: {len(btc_df_aligned)} | 山寨币数据量: {len(alt_df_aligned)}")
            logger.debug(f"币种: {coin} | {timeframe}/{period} 数据详情 | BTC: {btc_df.head()}, length: {len(btc_df)} | 山寨币: {alt_df.head()}, length: {len(alt_df)}")
            return None
        
        return btc_df_aligned, alt_df_aligned
    
    def _analyze_single_combination(self, coin: str, timeframe: str, period: str) -> tuple | None:
        """
        分析单个 timeframe/period 组合
        
        Returns:
            成功返回 (correlation, timeframe, period, tau_star)，失败返回 None
        """
        btc_df = self._get_btc_data(timeframe, period)
        if btc_df is None:
            return None
        
        alt_df = self._get_alt_data(coin, period, timeframe, coin)
        if alt_df is None:
            return None
        
        # 对齐和验证数据
        aligned_data = self._align_and_validate_data(btc_df, alt_df, coin, timeframe, period)
        if aligned_data is None:
            return None
        btc_df_aligned, alt_df_aligned = aligned_data
        
        tau_star, _, related_matrix = self.find_optimal_delay(
            btc_df_aligned['return'].values, 
            alt_df_aligned['return'].values
        )
        logger.debug(f"分析结果 | timeframe: {timeframe} | period: {period} | tau_star: {tau_star} | 相关系数: {related_matrix:.4f}")
        
        return (related_matrix, timeframe, period, tau_star)
    
    def _detect_anomaly_pattern(self, results: list) -> tuple[bool, float]:
        """
        检测异常模式：短期低相关但长期高相关
        
        异常模式判断阈值：
        - 长期相关系数 > LONG_TERM_CORR_THRESHOLD：长期与BTC有较强跟随性
        - 短期相关系数 < SHORT_TERM_CORR_THRESHOLD：短期存在明显滞后
        - 差值 > CORR_DIFF_THRESHOLD：短期和长期差异足够显著
        
        Returns:
            (is_anomaly, diff_amount): 是否异常模式、相关系数差值
        """
        short_periods = ['1d']
        long_periods = ['7d', '30d']
        
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
        
        logger.info(f"发现异常币种 | 交易所: {self.exchange_name} | 币种: {coin} | 差值: {diff_amount:.2f}")
        
        # 飞书消息内容
        content = f"{self.exchange_name}\n\n{coin} 相关系数分析结果\n{df_results.to_string(index=False)}\n"
        content += f"\n差值: {diff_amount:.2f}"
        logger.debug(f"详细分析结果:\n{df_results.to_string(index=False)}")
        
        # 只有在 lark_hook 有效时才发送飞书通知
        if self.lark_hook:
            sender(content, self.lark_hook)
        else:
            logger.warning(f"飞书通知未发送（LARKBOT_ID 未配置）| 币种: {coin}")
    
    def one_coin_analysis(self, coin: str) -> bool:
        """
        分析单个币种与BTC的相关系数，识别异常模式
        
        Args:
            coin: 币种交易对名称，如 "ETH/USDC:USDC"
        
        Returns:
            是否发现异常模式
        """
        results = []
        first_combination_checked = False
        
        for timeframe in self.timeframes:
            for period in self.periods:
                # 对于第一个组合，先检查数据是否存在
                if not first_combination_checked:
                    first_combination_checked = True
                    # 尝试获取第一个组合的数据，检查是否为空
                    alt_df = self._get_alt_data(coin, period, timeframe, coin)
                    if alt_df is None:
                        # 数据不存在，提前退出所有组合
                        logger.warning(f"币种数据不存在（第一个组合检查无数据），跳过后续所有组合 | 币种: {coin} | {timeframe}/{period}")
                        return False
                
                result = self._safe_execute(
                    self._analyze_single_combination,
                    coin, timeframe, period,
                    error_msg=f"处理 {coin} 的 {timeframe}/{period} 时发生异常"
                )
                
                if result is not None:
                    results.append(result)
        
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
            # 计算相关系数统计信息
            corrs = [r[0] for r in valid_results]
            min_corr = min(corrs) if corrs else 0
            max_corr = max(corrs) if corrs else 0
            logger.debug(f"常规数据 | 币种: {coin} | 相关系数范围: {min_corr:.4f} ~ {max_corr:.4f}")
            return False
    
    def run(self):
        """分析交易所中所有USDC永续合约交易对"""
        logger.info(f"启动分析器 | 交易所: {self.exchange_name} | "
                    f"时间周期: {self.timeframes} | 数据周期: {self.periods}")
        
        all_coins = self.exchange.load_markets()
        usdc_coins = [c for c in all_coins if '/USDC:USDC' in c and c != self.btc_symbol]
        total = len(usdc_coins)
        anomaly_count = 0
        skip_count = 0
        start_time = time.time()
        
        logger.info(f"发现 {total} 个 USDC 永续合约交易对")
        
        # 进度里程碑：25%, 50%, 75%, 100%
        milestones = {max(1, int(total * p)) for p in [0.25, 0.5, 0.75, 1.0]}
        
        for idx, coin in enumerate(usdc_coins, 1):
            logger.debug(f"检查币种: {coin}")
            
            result = self._safe_execute(
                self.one_coin_analysis,
                coin,
                error_msg=f"分析币种 {coin} 时发生错误"
            )
            if result is True:
                anomaly_count += 1
            elif result is None:
                skip_count += 1
            
            # 在里程碑位置打印进度
            if idx in milestones:
                logger.info(f"分析进度: {idx}/{total} ({idx * 100 // total}%)")
            
            # 币种之间的间隔：增加到 2 秒，避免触发 Hyperliquid 的限流
            time.sleep(2)
        
        elapsed = time.time() - start_time
        logger.info(
            f"分析完成 | 交易所: {self.exchange_name} | "
            f"总数: {total} | 异常: {anomaly_count} | 跳过: {skip_count} | "
            f"耗时: {elapsed:.1f}s | 平均: {elapsed/total:.2f}s/币种"
        )


if __name__ == "__main__":
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")
    analyzer.run()
