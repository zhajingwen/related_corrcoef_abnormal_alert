# spurious_te_signal.py
# 虚假传递熵信号分析工具
# 功能：分析山寨币与BTC的皮尔逊相关系数，识别存在时间差套利空间的异常币种
# 原理：通过计算不同时间周期和延迟下的相关系数，找出短期高相关但长期低相关的币种
#       这类币种存在锚定BTC价格走势的时间差套利机会

from enum import Flag
import ccxt  # 加密货币交易所API库
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from retry import retry  # 重试装饰器，用于网络请求失败时自动重试

# 配置日志系统
# 设置日志级别为INFO，输出格式包含时间、名称、级别和消息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SpuriousTEAnalyzer:
    """
    虚假传递熵分析器
    
    核心功能：
    1. 下载山寨币和BTC的历史K线数据
    2. 计算不同时间周期（1m, 5m等）和不同数据周期（1d, 7d, 30d, 60d）下的相关系数
    3. 通过寻找最优延迟（tau*）来识别币种与BTC的滞后关系
    4. 识别异常模式：短期高相关但长期低相关的币种，这类币种可能存在套利机会
    
    异常模式定义：
    - 模式1：1分钟级别K线最大相关系数 > 0.4，但长期（60天）最大相关系数 < 0.05
      表示短期存在很大滞后性，但长期跟随性弱，存在时间差套利空间
    - 模式2：1分钟级别K线最大相关系数 < 0.11，且长期最大相关系数 < 0.05
      表示币种与BTC相关性极低，可能存在独立走势或异常行为
    """
    
    def __init__(self, exchange_name="kucoin", timeout=30000, default_timeframes=None, default_periods=None):
        """
        初始化分析器
        
        Args:
            exchange_name (str): 交易所名称，默认为 "kucoin"
                                支持ccxt库支持的所有交易所
            timeout (int): 请求超时时间（毫秒），默认30000ms（30秒）
            default_timeframes (list): 默认时间周期列表，K线的颗粒度
                                      None时默认为 ["1m", "5m"]
                                      例如："1m"表示1分钟K线，"5m"表示5分钟K线
            default_periods (list): 默认数据周期列表，K线的覆盖范围
                                   None时默认为 ["1d", "7d", "30d", "60d"]
                                   例如："1d"表示最近1天的数据，"60d"表示最近60天的数据
        
        属性说明：
            exchange: ccxt交易所实例，用于获取市场数据
            timeframes: 要分析的时间周期列表
            periods: 要分析的数据周期列表
            btc_symbol: BTC交易对名称，固定为 "BTC/USDT"
            btc_df_cache: BTC数据缓存字典，避免重复下载相同的数据
                          key格式：(timeframe, period)，例如：("1m", "60d")
                          value: 对应的BTC DataFrame数据
        """
        # 动态获取交易所实例，使用反射机制根据名称创建对应的交易所对象
        self.exchange = getattr(ccxt, exchange_name)({"timeout": timeout})
        # 设置默认时间周期：1分钟和5分钟K线
        self.timeframes = default_timeframes or ["1m", "5m"]
        # 设置默认数据周期：1天、7天、30天、60天
        self.periods = default_periods or ["1d", "7d", "30d", "60d"]
        # BTC的交易对名称，作为基准货币
        self.btc_symbol = "BTC/USDT"
        # 缓存BTC各个颗粒度、各个周期级别的数据
        # 由于BTC数据对所有山寨币都相同，缓存可以大幅减少API调用次数
        self.btc_df_cache = {}  # 缓存字典，key为 (timeframe, period) 元组
    
    @staticmethod
    def _period_to_bars(period: str, timeframe: str) -> int:
        """
        将时间周期和数据周期转换为K线总条数（bars）
        
        计算公式：
        - 每天K线数 = 24小时 * 60分钟 / 每个K线的分钟数
        - 总K线数 = 天数 * 每天K线数
        
        示例：
        - period="60d", timeframe="5m" → 60天 * (24*60/5) = 60 * 288 = 17280条K线
        - period="1d", timeframe="1m" → 1天 * (24*60/1) = 1440条K线
        
        Args:
            period (str): 数据周期（K线的覆盖范围），格式为 "Xd"，X为天数
                         例如："1d"表示1天，"60d"表示60天
            timeframe (str): 时间周期（K线的颗粒度），格式为 "Xm"，X为分钟数
                           例如："1m"表示1分钟K线，"5m"表示5分钟K线
        
        Returns:
            int: 统计周期内的K线总条数（bars）
        """
        # 提取天数：去掉末尾的'd'字符并转换为整数
        days = int(period.rstrip('d'))
        # 提取分钟数：去掉末尾的'm'字符并转换为整数
        timeframe_minutes = int(timeframe.rstrip('m'))
        # 计算每天有多少根K线：24小时 * 60分钟 / 每根K线的分钟数
        bars_per_day = int(24 * 60 / timeframe_minutes)
        # 返回总K线数
        return days * bars_per_day
    
    @retry(tries=10, delay=5, backoff=2)
    def download_ccxt_data(self, symbol: str, period: str, timeframe: str) -> pd.DataFrame:
        """
        从交易所下载指定交易对的OHLCV（开高低收成交量）历史数据
        
        功能说明：
        1. 计算需要下载的K线总数
        2. 分批次下载数据（每次最多1500条，因为交易所API通常有限制）
        3. 处理数据格式转换和时间戳对齐
        4. 计算收益率和美元成交量
        
        重试机制：
        - 使用@retry装饰器，最多重试10次
        - 每次重试间隔5秒，指数退避（backoff=2）
        - 函数内部还有额外的5次重试机制，用于处理单次API调用失败
        
        Args:
            symbol (str): 交易对名称，例如 "BTC/USDT"、"ETH/USDT"
            period (str): 数据周期，例如 "60d"（60天）
            timeframe (str): K线时间周期，例如 "5m"（5分钟K线）
        
        Returns:
            pd.DataFrame: 包含以下列的DataFrame：
                - Timestamp: 时间戳（作为索引）
                - Open: 开盘价
                - High: 最高价
                - Low: 最低价
                - Close: 收盘价
                - Volume: 成交量（以基础货币为单位）
                - return: 收益率（相邻K线的价格变化百分比）
                - volume_usd: 美元成交量（Volume * Close）
        
        注意：
            - 如果数据为空，返回包含所有必要列的空DataFrame
            - 时间戳转换为本地时区（去除UTC时区信息）
            - 数据按时间戳升序排列
        """
        # 计算目标K线总数
        target_bars = self._period_to_bars(period, timeframe)
        # 计算每根K线对应的毫秒数（timeframe为分钟级）
        # 例如：5m → 5 * 60 * 1000 = 300000毫秒
        ms_per_bar = int(timeframe.rstrip('m')) * 60 * 1000
        # 获取当前时间戳（毫秒）
        now_ms = self.exchange.milliseconds()
        # 计算起始时间戳：当前时间 - 目标K线数 * 每根K线的毫秒数
        since = now_ms - target_bars * ms_per_bar

        # 存储所有下载的K线数据
        all_rows = []
        # 已获取的K线数量
        fetched = 0
        
        # 循环下载数据，直到获取足够的数据或没有更多数据
        while True:
            # 带重试机制的OHLCV数据抓取
            # 内部重试机制：最多尝试5次，每次失败后等待时间递增
            last_exc = None
            for attempt in range(5):
                try:
                    # 调用交易所API获取OHLCV数据
                    # limit=1500：每次最多获取1500条K线（交易所API限制）
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
                    last_exc = None
                    break  # 成功获取数据，跳出重试循环
                except Exception as e:
                    # 记录异常，等待后重试
                    last_exc = e
                    # 指数退避：第1次等待1.5秒，第2次等待3秒，第3次等待4.5秒...
                    time.sleep(1.5 * (attempt + 1))
            
            # 如果5次重试都失败，抛出最后一个异常
            if last_exc is not None:
                raise last_exc
            
            # 如果没有获取到数据，退出循环
            if not ohlcv:
                break
            
            # 将本次获取的数据添加到总列表
            all_rows.extend(ohlcv)
            fetched += len(ohlcv)
            
            # 更新起始时间戳：使用最后一条K线的时间戳 + 1根K线的时长
            # 这样下次请求会从下一条K线开始
            since = ohlcv[-1][0] + ms_per_bar
            
            # 如果获取的数据少于1500条（说明已经到历史数据边界）
            # 或者已经获取了足够的数据，则退出循环
            if len(ohlcv) < 1500 or fetched >= target_bars:
                break

        # 如果没有获取到任何数据，返回空DataFrame（包含必要的列）
        if not all_rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "return", "volume_usd"])

        # 将原始数据转换为DataFrame
        # OHLCV格式：[timestamp, open, high, low, close, volume]
        df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        
        # 将时间戳从毫秒转换为datetime对象
        # unit="ms": 输入是毫秒时间戳
        # utc=True: 先转换为UTC时区
        # dt.tz_convert(None): 再转换为本地时区（去除时区信息）
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        
        # 将时间戳设置为索引，并按时间升序排列
        df = df.set_index("Timestamp").sort_index()
        
        # 计算收益率：相邻K线收盘价的百分比变化
        # pct_change(): 计算百分比变化，第一个值为NaN（因为没有前一个值）
        # fillna(0): 将NaN填充为0（第一个K线的收益率为0）
        df['return'] = df['Close'].pct_change().fillna(0)
        
        # 计算美元成交量：成交量（基础货币） * 收盘价（USDT）
        df['volume_usd'] = df['Volume'] * df['Close']
        
        return df
    
    @staticmethod
    def find_optimal_delay(btc_ret, alt_ret, max_lag=48):
        """
        寻找最优延迟 τ*（tau star）
        
        功能说明：
        通过计算不同延迟（lag）下BTC和山寨币收益率的相关系数，找出使相关系数最大的延迟值。
        这个延迟值表示山寨币相对于BTC的滞后时间。
        
        算法原理：
        - 对于每个延迟lag（0到max_lag），计算BTC[t]和ALT[t+lag]的相关系数
        - lag=0：计算BTC和ALT同时刻的相关系数（同步相关性）
        - lag>0：计算BTC[t]和ALT[t+lag]的相关系数（滞后相关性）
          - 如果lag=5，表示ALT滞后BTC 5个时间单位
          - 即：BTC在t时刻的价格变化，会影响ALT在t+5时刻的价格
        
        最优延迟的意义：
        - tau_star表示使相关系数最大的延迟值
        - 如果tau_star>0，说明ALT滞后于BTC，存在时间差套利机会
        - 如果tau_star=0，说明ALT与BTC同步，没有明显滞后
        
        Args:
            btc_ret (np.ndarray): BTC收益率数组，形状为(n,)
            alt_ret (np.ndarray): 山寨币收益率数组，形状为(n,)
            max_lag (int): 最大延迟值，默认48
                         对于1分钟K线，max_lag=48表示最多滞后48分钟
                         对于5分钟K线，max_lag=48表示最多滞后240分钟（4小时）
        
        Returns:
            tuple: (tau_star, corrs, max_related_matrix)
                - tau_star (int): 最优延迟值，使相关系数最大的lag值
                - corrs (list): 所有延迟值对应的相关系数列表，长度为max_lag+1
                               索引i对应lag=i时的相关系数
                - max_related_matrix (float): 最大相关系数值，范围[-1, 1]
                                            接近1表示强正相关，接近-1表示强负相关
                                            接近0表示无相关性
        
        示例：
            如果tau_star=10，max_related_matrix=0.8，表示：
            - ALT滞后BTC 10个时间单位时，相关系数最大（0.8）
            - 即BTC在t时刻的价格变化，会在10个时间单位后影响ALT的价格
        """
        # 存储每个延迟值对应的相关系数
        corrs = []
        # 生成延迟值列表：[0, 1, 2, ..., max_lag]
        lags = list(range(0, max_lag + 1))
        
        # 遍历每个延迟值，计算对应的相关系数
        for lag in lags:
            if lag > 0:
                # ALT滞后BTC的情况：验证BTC[t]是否影响ALT[t+lag]
                # 例如lag=5：比较BTC的前n-5个数据与ALT的后n-5个数据
                # 这样对齐后，BTC[i]对应ALT[i+5]，表示ALT滞后5个时间单位
                x = btc_ret[:-lag]  # BTC的前n-lag个数据点
                y = alt_ret[lag:]   # ALT的后n-lag个数据点（跳过前lag个）
            elif lag == 0:
                # lag=0：计算同步相关性，使用全样本
                # 注意：这里x和y的顺序可能影响相关系数的符号
                # 但相关系数是对称的，所以顺序不影响绝对值
                x = alt_ret
                y = btc_ret
            
            # 确保两个数组长度一致（取最小值）
            m = min(len(x), len(y))
            
            # 如果数据点太少（少于10个），无法计算可靠的相关系数
            # 将相关系数设为-1（无效值）
            if m < 10:
                corrs.append(-1)
                continue
            
            # 计算皮尔逊相关系数
            # np.corrcoef返回2x2的相关系数矩阵：
            # [[corr(x,x), corr(x,y)],
            #  [corr(y,x), corr(y,y)]]
            # [0,1]位置就是x和y的相关系数
            related_matrix = np.corrcoef(x[:m], y[:m])[0, 1]
            corrs.append(related_matrix)
        
        # 找出使相关系数最大的延迟值（最优延迟）
        tau_star = lags[np.argmax(corrs)]
        # 获取最大相关系数值
        max_related_matrix = max(corrs)
        
        return tau_star, corrs, max_related_matrix
    
    @staticmethod
    def compute_spurious_te(btc_ret, alt_ret, delay, k=3):
        """
        计算虚假传递熵（Spurious Transfer Entropy）: T_{ALT → BTC}(τ)
        
        传递熵（TE）定义：
        TE(Y→X) = I(X_t; Y_{t-τ} | X_{t-1})
        表示在已知X_{t-1}的条件下，Y_{t-τ}对X_t的额外信息量
        
        虚假传递熵的含义：
        - 如果ALT对BTC存在虚假的传递熵，说明ALT的价格变化"预测"了BTC的未来价格
        - 但实际上这可能是由于两者都受到共同因素影响，而非真正的因果关系
        - 这种虚假的TE信号可能揭示套利机会
        
        计算方法：
        1. 将连续收益率离散化为3个区间（分位数分箱）
        2. 计算条件互信息 I(X_t; Y_{t-τ} | X_{t-1})
        3. 使用频率估计概率分布，然后计算KL散度
        
        数学公式：
        TE = Σ p(x_t, x_{t-1}, y_{t-τ}) * log2(p(x_t, x_{t-1}, y_{t-τ}) * p(x_{t-1}) / 
                                              (p(x_{t-1}, y_{t-τ}) * p(x_t, y_{t-τ})))
        
        Args:
            btc_ret (np.ndarray): BTC收益率数组
            alt_ret (np.ndarray): 山寨币收益率数组
            delay (int): 延迟值τ，表示ALT滞后BTC的时间单位
            k (int): 参数，当前未使用（保留用于未来扩展），默认3
        
        Returns:
            float: 传递熵值，范围[0, +∞)
                  - 值越大，表示ALT对BTC的信息传递越强
                  - 0表示没有信息传递
                  - 如果数据不足或计算失败，返回0.0
        
        注意：
            - 需要至少200个数据点才能进行可靠计算
            - 使用分位数分箱将连续值离散化，避免极端值影响
            - 添加小量epsilon（1e-12）避免log(0)的情况
        """
        # 采用离散近似的TE估计：TE ≈ I(X_t; Y_{t-τ} | X_{t-1})
        # 其中X_t表示BTC在t时刻的收益率，Y_{t-τ}表示ALT在t-τ时刻的收益率
        
        # 数据量检查：需要足够的数据点才能进行可靠计算
        if len(btc_ret) < 200 or len(alt_ret) < 200 or delay < 0:
            return 0.0
        
        try:
            # 分箱数量：将连续收益率离散化为3个区间
            num_bins = 3
            
            # 分位数分箱函数：将连续值转换为离散值
            # 使用分位数而不是等间距分箱，可以避免极端值的影响
            def discretize(x):
                """
                将连续收益率离散化为0, 1, 2三个值
                使用33%和67%分位数作为分箱边界
                """
                # 计算分位数：33%和67%分位数
                qs = np.nanquantile(x, [1/num_bins, 2/num_bins])
                # np.digitize返回每个值所属的区间索引（0, 1, 2）
                return np.digitize(x, qs)

            # 将BTC和ALT的收益率离散化
            xb = discretize(btc_ret)  # BTC离散化后的值
            yb = discretize(alt_ret)  # ALT离散化后的值

            # 对齐时间序列：X_t, X_{t-1}, Y_{t-τ}
            # X_t: BTC在t时刻的值（从索引1+delay开始）
            # X_p: BTC在t-1时刻的值（从索引delay开始，到倒数第二个）
            # Y_p: ALT在t-τ时刻的值（从开头开始，到倒数1+delay个）
            X_t = xb[1+delay:]      # BTC当前时刻
            X_p = xb[delay:-1]      # BTC前一时刻
            Y_p = yb[:-1-delay]     # ALT延迟τ时刻

            # 确保三个数组长度一致
            n = min(len(X_t), len(X_p), len(Y_p))
            # 如果对齐后的数据点太少，无法进行可靠计算
            if n <= 100:
                return 0.0
            # 截取相同长度的数据
            X_t, X_p, Y_p = X_t[:n], X_p[:n], Y_p[:n]

            # 计算条件互信息 I(X_t; Y_p | X_p)
            # 需要统计以下概率分布：
            # - p(x_t, x_{t-1}, y_{t-τ}): 三维联合概率
            # - p(x_t, y_{t-τ}): 二维边际概率
            # - p(x_{t-1}, y_{t-τ}): 二维边际概率
            # - p(x_{t-1}): 一维边际概率
            
            # 确定概率分布数组的维度
            max_x = int(max(X_t.max(), X_p.max())) + 1  # X的最大值+1（因为索引从0开始）
            max_y = int(Y_p.max()) + 1                   # Y的最大值+1
            
            # 初始化概率分布数组
            p_xyz = np.zeros((max_x, max_x, max_y), dtype=np.float64)  # p(x_t, x_{t-1}, y_{t-τ})
            p_xz = np.zeros((max_x, max_y), dtype=np.float64)          # p(x_t, y_{t-τ})
            p_yz = np.zeros((max_x, max_y), dtype=np.float64)          # p(x_{t-1}, y_{t-τ})
            p_z = np.zeros((max_x,), dtype=np.float64)                 # p(x_{t-1})

            # 统计频率：遍历所有对齐的数据点，统计联合频率
            for a, b, c in zip(X_t, X_p, Y_p):
                # a: X_t的值，b: X_p的值，c: Y_p的值
                p_xyz[a, b, c] += 1  # 三维联合频率
                p_xz[a, c] += 1      # (X_t, Y_p)联合频率
                p_yz[b, c] += 1      # (X_p, Y_p)联合频率
                p_z[b] += 1          # X_p边际频率

            # 将频率转换为概率（归一化）
            p_xyz /= n
            p_xz /= n
            p_yz /= n
            p_z /= n

            # 计算传递熵：TE = Σ p(x_t, x_{t-1}, y_{t-τ}) * log2(...)
            eps = 1e-12  # 小量，避免log(0)的情况
            te = 0.0
            
            # 使用numpy的nditer遍历三维数组的所有元素
            it = np.nditer(p_xyz, flags=['multi_index'])
            while not it.finished:
                pabc = float(it[0])  # p(x_t=a, x_{t-1}=b, y_{t-τ}=c)
                
                # 只计算概率大于0的情况（避免log(0)）
                if pabc > 0:
                    a, b, c = it.multi_index  # 获取当前索引
                    
                    # 计算KL散度公式中的分子和分母
                    # 公式：log2(p(x_t, x_{t-1}, y_{t-τ}) * p(x_{t-1}) / 
                    #            (p(x_{t-1}, y_{t-τ}) * p(x_t, y_{t-τ})))
                    num = pabc * p_z[b]                    # 分子：p(x_t, x_{t-1}, y_{t-τ}) * p(x_{t-1})
                    den = (p_yz[b, c] * p_xz[a, c])        # 分母：p(x_{t-1}, y_{t-τ}) * p(x_t, y_{t-τ})
                    
                    # 只有当分子和分母都大于0时才计算
                    if num > 0 and den > 0:
                        # 累加传递熵值
                        te += pabc * np.log2((num + eps) / (den + eps))
                
                it.iternext()  # 移动到下一个元素

            # 返回传递熵值（确保非负）
            return max(float(te), 0.0)
        except Exception:
            # 如果计算过程中出现任何异常，返回0.0
            return 0.0
    
    @staticmethod
    def generate_signal(te_value, threshold=0.05):
        """
        根据传递熵值生成交易信号
        
        功能说明：
        根据计算得到的传递熵（TE）值，判断是否触发套利信号。
        如果TE值超过阈值，说明存在明显的信息传递，可能有机会进行延迟套利。
        
        Args:
            te_value (float): 传递熵值，范围[0, +∞)
            threshold (float): 触发信号的阈值，默认0.05
                              - 如果TE值 > threshold，生成"ENTER"信号
                              - 如果TE值 <= threshold，生成"HOLD"信号
        
        Returns:
            str: 交易信号字符串
                - "ENTER: 延迟套利信号触发！": TE值超过阈值，可能存在套利机会
                - "HOLD: 虚假 TE 不足": TE值未超过阈值，暂不操作
        
        注意：
            当前代码中此方法未被调用，可能是预留功能或未来扩展使用
        """
        if te_value > threshold:
            return "ENTER: 延迟套利信号触发！"
        else:
            return "HOLD: 虚假 TE 不足"
    
    def one_coin_analysis(self, coin: str):
        """
        分析单个币种与BTC的相关系数，识别异常模式
        
        功能说明：
        1. 遍历所有时间周期（timeframes）和数据周期（periods）的组合
        2. 下载BTC和该币种的历史数据
        3. 计算每个组合下的最优延迟和最大相关系数
        4. 识别异常模式：短期高相关但长期低相关的币种
        
        异常模式识别：
        - 模式1：短期（1分钟K线）最大相关系数 > 0.4，但长期（60天）最大相关系数 < 0.05
          含义：短期存在很大滞后性，但长期跟随性弱，存在时间差套利空间
        - 模式2：短期最大相关系数 < 0.11，且长期最大相关系数 < 0.05
          含义：币种与BTC相关性极低，可能存在独立走势或异常行为
        
        输出：
        - 如果检测到异常模式，会输出详细的相关系数分析结果表格
        - 表格包含：最大相关系数、时间周期、数据周期、最优延迟
        
        Args:
            coin (str): 币种交易对名称，例如 "KCS/USDT"、"ETH/USDT"
        
        处理流程：
        1. 遍历所有timeframe和period组合
        2. 下载BTC数据（使用缓存避免重复下载）
        3. 下载山寨币数据
        4. 对齐时间索引，确保数据时间一致
        5. 计算最优延迟和最大相关系数
        6. 收集所有结果并按相关系数排序
        7. 判断是否为异常模式，如果是则输出结果
        """
        # 存储所有组合的最大相关系数
        # key: 最大相关系数, value: (timeframe, period, tau_star) 元组
        max_related_matrix_list = {}
        
        # 遍历所有时间周期和数据周期的组合
        for timeframe in self.timeframes:
            for period in self.periods:
                logger.info(f"正在下载 {coin} 的 {timeframe}、{period} 数据...")
                
                # 缓存BTC数据，避免重复下载（因为BTC数据对所有币种都相同）
                cache_key = (timeframe, period)
                if cache_key not in self.btc_df_cache:
                    # 如果缓存中没有，下载BTC数据并存入缓存
                    self.btc_df_cache[cache_key] = self.download_ccxt_data(
                        self.btc_symbol, period=period, timeframe=timeframe
                    )
                
                # 必须使用 .copy()，避免后续操作（如 loc 索引）修改缓存的数据
                # 如果直接使用缓存数据，修改会影响后续币种的分析
                btc_df = self.btc_df_cache[cache_key].copy()
                
                # 下载山寨币数据
                alt_df = self.download_ccxt_data(coin, period=period, timeframe=timeframe)
                
                # 对齐时间索引：只保留BTC和山寨币都有的时间点
                # 这样可以确保两个时间序列的时间一致
                common_idx = btc_df.index.intersection(alt_df.index)
                btc_df, alt_df = btc_df.loc[common_idx], alt_df.loc[common_idx]
                
                # 检查数据是否为空或缺少必要的列
                if len(btc_df) == 0 or len(alt_df) == 0:
                    logger.warning(f"  警告: {coin} 的 {timeframe}/{period} 数据为空，跳过...")
                    continue
                if 'return' not in btc_df.columns or 'return' not in alt_df.columns:
                    logger.warning(f"  警告: {coin} 的 {timeframe}/{period} 数据缺少 'return' 列，跳过...")
                    continue
                
                # 提取收益率数组
                btc_ret = btc_df['return'].values
                alt_ret = alt_df['return'].values
                
                # 找最优延迟（单位：分钟级 bars）
                # tau_star: 最优延迟值
                # corr_curve: 所有延迟值对应的相关系数列表（当前未使用）
                # max_related_matrix: 最大相关系数值
                tau_star, corr_curve, max_related_matrix = self.find_optimal_delay(btc_ret, alt_ret)
                logger.info(f'timeframe: {timeframe}, period: {period}, tau_star: {tau_star}, max_related_matrix: {max_related_matrix}')

                # 存储结果：使用最大相关系数作为key
                # 注意：如果多个组合有相同的最大相关系数，后面的会覆盖前面的
                # 但由于我们按相关系数排序，这个影响不大
                max_related_matrix_list[max_related_matrix] = (timeframe, period, tau_star)

        # 按最大相关系数降序排序
        # 排序后，第一行是相关系数最大的组合，最后一行是相关系数最小的组合
        max_related_matrix_list = sorted(max_related_matrix_list.items(), key=lambda x: x[0], reverse=True)
        
        # 转换为pandas DataFrame，方便查看和输出
        df_results = pd.DataFrame([
            {
                '最大相关系数': max_corr,
                '时间周期': timeframe,
                '数据周期': period,
                '最优延迟': tau_star
            }
            for max_corr, (timeframe, period, tau_star) in max_related_matrix_list
        ])
        
        # 判断是否需要输出结果（是否为异常模式）
        print_status = False
        
        # 获取"最大相关系数"列的第一行和最后一行的值
        if len(df_results) > 0:
            first_max_corr = df_results.iloc[0]['最大相关系数']  # 最大相关系数（短期）
            last_max_corr = df_results.iloc[-1]['最大相关系数']  # 最小相关系数（长期）
            
            # 异常模式1：第一行最大相关系数大于0.4，最后一行最大相关系数小于0.05
            # 这个数据状态表示1分钟级别的K线存在很大的滞后性，但是长期又表现出跟随的特性
            # 那这种就存在锚定BTC价格走势的时间差套利空间
            if first_max_corr > 0.4 and last_max_corr < 0.05:
                print_status = True
            
            # 异常模式2：第一行最大相关系数小于0.11，最后一行最大相关系数小于0.05
            # 表示币种与BTC相关性极低，可能存在独立走势或异常行为
            elif first_max_corr < 0.11 and last_max_corr < 0.05:
                print_status = True
            else:
                # 常规数据，不输出详细结果
                logger.info(f'常规数据：{coin}')
        
        # 如果是异常模式，输出详细的相关系数分析结果
        if print_status:
            # 格式化输出，使用分隔线使结果更清晰
            logger.info("\n" + "="*60)
            logger.info(f"{coin}相关系数分析结果")
            logger.info("="*60)
            logger.info(df_results.to_string(index=False))
            logger.info("="*60)
    
    def run(self, quote_currency="USDT"):
        """
        主运行方法，分析交易所中所有指定计价货币的交易对
        
        功能说明：
        1. 加载交易所的所有市场信息
        2. 筛选出指定计价货币的交易对（默认USDT）
        3. 对每个交易对进行分析，识别异常模式
        4. 在每次分析之间添加延迟，避免API请求过于频繁
        
        Args:
            quote_currency (str): 计价货币，默认为 "USDT"
                                只分析以该货币计价的交易对
                                例如："USDT"会分析所有XXX/USDT交易对
        
        处理流程：
        1. 调用exchange.load_markets()加载所有市场信息
        2. 遍历所有交易对
        3. 检查交易对的quote字段是否匹配quote_currency
        4. 对匹配的交易对调用one_coin_analysis()进行分析
        5. 每次分析后等待1秒，避免触发API限流
        
        输出：
        - 对于检测到异常模式的币种，会输出详细的相关系数分析结果
        - 对于常规币种，只输出简单的日志信息
        
        注意：
            - 分析所有币种可能需要较长时间，取决于交易所的交易对数量
            - 使用BTC数据缓存可以大幅减少API调用次数
            - 如果交易所API有速率限制，可能需要调整sleep时间
        """
        # 加载交易所的所有市场信息
        # 返回一个字典，key是交易对名称（如"BTC/USDT"），value是市场信息字典
        all_coins = self.exchange.load_markets()
        
        # 遍历所有交易对
        for coin in all_coins:
            coin_item = all_coins[coin]
            
            # 只分析指定计价货币的交易对
            # coin_item['quote']是计价货币，例如"USDT"、"BTC"等
            if coin_item['quote'] != quote_currency:
                continue  # 跳过不匹配的交易对
            
            # 分析当前币种
            self.one_coin_analysis(coin)
            
            # 等待1秒，避免API请求过于频繁
            # 这有助于避免触发交易所的API速率限制
            time.sleep(1)


# ================== 运行 ==================
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = SpuriousTEAnalyzer()
    # 运行分析，分析所有USDT交易对
    analyzer.run()
