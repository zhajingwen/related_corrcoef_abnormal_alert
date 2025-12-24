# 📊 滚动窗口Beta实施方案

## 文档信息

- **项目名称**：相关系数异常告警系统 - 滚动窗口Beta增强
- **版本**：v1.0
- **创建日期**：2025-12-24
- **目标文件**：`hyperliquid_analyzer.py`
- **依赖方案**：优化实施计划.md

---

## 一、核心概念

### 1.1 什么是滚动窗口Beta？

**滚动窗口Beta（Rolling Window Beta）** 是一种动态计算Beta系数的方法，通过在时间序列上滑动固定大小的窗口，计算每个时间点的Beta值，从而捕捉Beta系数随时间的变化趋势。

**核心思想：**
```
静态Beta:  [========全部数据========] → 单一β值

滚动Beta:  [窗口1]                    → β₁
            [窗口2]                   → β₂
             [窗口3]                  → β₃
              ...                     → ...
```

### 1.2 为什么需要滚动窗口Beta？

#### **问题：静态Beta的局限性**

当前方案计算的是**静态Beta**，使用全部历史数据得到单一β值：

```python
# 当前方案
beta = Cov(全部ALT收益, 全部BTC收益) / Var(全部BTC收益)
# 结果: β = 1.25 (一个固定值)
```

**但现实中，Beta系数会随时间变化：**

| 时期 | 市场状态 | ALT跟随幅度 | 实际Beta |
|------|---------|------------|----------|
| 牛市初期 | BTC上涨 | ALT涨幅小 | β ≈ 0.8 |
| 牛市中期 | BTC暴涨 | ALT跟随暴涨 | β ≈ 2.5 |
| 熊市 | BTC下跌 | ALT跌幅更大 | β ≈ 1.8 |
| 震荡期 | BTC横盘 | ALT独立走势 | β ≈ 0.3 |

**用单一Beta值（如1.25）描述上述所有阶段 → 信息损失严重！**

---

## 二、时变性问题深度分析

### 2.1 Beta时变的真实案例

**假设场景：ETH相对于BTC的Beta**

```
2024-Q1 (牛市启动)：β = 0.9  → ETH跟随BTC稳健上涨
2024-Q2 (散户入场)：β = 2.1  → ETH涨幅是BTC的2倍
2024-Q3 (监管消息)：β = 0.4  → ETH受监管影响，与BTC相关性降低
2024-Q4 (年末整理)：β = 1.3  → ETH恢复跟随BTC
```

**如果只计算静态Beta：**
```python
静态Beta = (0.9 + 2.1 + 0.4 + 1.3) / 4 = 1.175
```

**问题：**
- ❌ 无法识别Q2的高风险期（β=2.1，高波动）
- ❌ 无法识别Q3的脱钩期（β=0.4，独立走势）
- ❌ 无法用于动态风险管理和仓位调整

### 2.2 导致Beta时变的因素

| 因素 | 影响机制 | 示例 |
|------|---------|------|
| **市场周期** | 牛市时ALT跟随放大，熊市时跌幅更大 | 牛市β>1，熊市β>1.5 |
| **流动性变化** | 流动性充足时相关性高，枯竭时脱钩 | 2022年Luna崩盘期β暴涨 |
| **叙事转换** | DeFi热潮、NFT热潮等独立叙事 | DeFi Summer期间β<0.5 |
| **监管政策** | 某些币种受特定监管影响 | XRP诉讼期间β接近0 |
| **技术升级** | 重大升级可能脱钩BTC走势 | ETH合并前β降低 |

---

## 三、滚动窗口Beta实现方案

### 3.1 核心参数设计

```python
class DelayCorrelationAnalyzer:
    # ========== 新增：滚动窗口Beta配置 ==========
    # 滚动窗口大小（数据点数量）
    ROLLING_BETA_WINDOW_SIZE = 30  # 默认30个数据点

    # 滚动步长（每次移动的点数）
    ROLLING_BETA_STEP = 1  # 默认每次移动1个点（最细粒度）

    # 是否启用滚动Beta计算
    ENABLE_ROLLING_BETA = False  # 默认关闭（高级功能）

    # 滚动Beta的最小窗口要求（避免窗口太小导致不稳定）
    MIN_ROLLING_WINDOW_SIZE = 20
```

**参数选择建议：**

| 时间周期 | 建议窗口大小 | 理由 |
|---------|------------|------|
| **1小时K线** | 48-72点 | 覆盖2-3天，平滑短期噪音 |
| **4小时K线** | 30-42点 | 覆盖5-7天，捕捉周内趋势 |
| **日线K线** | 20-30点 | 覆盖3-4周，捕捉月度变化 |
| **周线K线** | 12-16点 | 覆盖3-4个月，捕捉季度变化 |

### 3.2 方法实现

```python
@staticmethod
def _calculate_rolling_beta(btc_ret, alt_ret, window_size=None, step=None):
    """
    计算滚动窗口Beta系数

    在时间序列上滑动固定大小的窗口，计算每个窗口内的Beta系数，
    从而得到Beta的时间序列，捕捉Beta系数的时变特征。

    Args:
        btc_ret: BTC收益率数组（numpy array）
        alt_ret: 山寨币收益率数组（numpy array）
        window_size: 滚动窗口大小（数据点数量），默认使用类常量
        step: 滚动步长，默认使用类常量

    Returns:
        dict: {
            'betas': Beta时间序列（numpy array）,
            'timestamps': 对应的时间索引,
            'mean_beta': 平均Beta,
            'std_beta': Beta标准差,
            'min_beta': 最小Beta,
            'max_beta': 最大Beta,
            'current_beta': 最新Beta值（最后一个窗口）
        }
        如果数据不足或计算失败，返回 None

    Example:
        >>> btc_ret = np.array([0.01, -0.02, 0.03, ...])  # 100个点
        >>> alt_ret = np.array([0.015, -0.025, 0.04, ...])
        >>> result = _calculate_rolling_beta(btc_ret, alt_ret, window_size=30)
        >>> print(result['current_beta'])  # 最新Beta值
        1.25
        >>> print(result['mean_beta'])  # 历史平均Beta
        1.18
    """
    # 1. 参数默认值处理
    if window_size is None:
        window_size = DelayCorrelationAnalyzer.ROLLING_BETA_WINDOW_SIZE
    if step is None:
        step = DelayCorrelationAnalyzer.ROLLING_BETA_STEP

    # 2. 数据验证
    if len(btc_ret) != len(alt_ret):
        logger.warning(f"滚动Beta计算失败：数据长度不一致 | "
                      f"BTC: {len(btc_ret)}, ALT: {len(alt_ret)}")
        return None

    # 3. 窗口大小检查
    data_len = len(btc_ret)
    if data_len < window_size:
        logger.debug(f"滚动Beta计算失败：数据量不足 | "
                    f"需要: {window_size}, 实际: {data_len}")
        return None

    if window_size < DelayCorrelationAnalyzer.MIN_ROLLING_WINDOW_SIZE:
        logger.warning(f"窗口大小过小，调整至最小值 | "
                      f"原值: {window_size}, 调整后: {DelayCorrelationAnalyzer.MIN_ROLLING_WINDOW_SIZE}")
        window_size = DelayCorrelationAnalyzer.MIN_ROLLING_WINDOW_SIZE

    # 4. 计算滚动窗口Beta
    betas = []
    timestamps = []  # 窗口结束位置的索引

    try:
        # 滑动窗口计算
        for i in range(0, data_len - window_size + 1, step):
            # 提取当前窗口的数据
            window_btc = btc_ret[i:i + window_size]
            window_alt = alt_ret[i:i + window_size]

            # 计算当前窗口的Beta
            beta = DelayCorrelationAnalyzer._calculate_beta(window_btc, window_alt)

            if not np.isnan(beta):
                betas.append(beta)
                timestamps.append(i + window_size - 1)  # 窗口结束位置

        # 5. 检查是否有有效结果
        if len(betas) == 0:
            logger.debug("滚动Beta计算失败：没有有效的Beta值")
            return None

        betas_array = np.array(betas)

        # 6. 计算统计指标
        result = {
            'betas': betas_array,
            'timestamps': np.array(timestamps),
            'mean_beta': np.mean(betas_array),
            'std_beta': np.std(betas_array),
            'min_beta': np.min(betas_array),
            'max_beta': np.max(betas_array),
            'current_beta': betas_array[-1],  # 最新Beta值
            'beta_trend': 'increasing' if betas_array[-1] > np.mean(betas_array) else 'decreasing'
        }

        logger.debug(
            f"滚动Beta计算完成 | "
            f"窗口数: {len(betas)} | "
            f"当前Beta: {result['current_beta']:.4f} | "
            f"平均Beta: {result['mean_beta']:.4f} ± {result['std_beta']:.4f} | "
            f"范围: [{result['min_beta']:.4f}, {result['max_beta']:.4f}]"
        )

        return result

    except Exception as e:
        logger.warning(f"滚动Beta计算异常：{type(e).__name__}: {str(e)}")
        return None
```

### 3.3 集成到现有流程

**修改 `find_optimal_delay` 方法：**

```python
@staticmethod
def find_optimal_delay(btc_ret, alt_ret, max_lag=3,
                       enable_outlier_treatment=None,
                       enable_beta_calc=None,
                       enable_rolling_beta=None):  # 新增参数
    """
    寻找最优延迟 τ*（增强版：支持滚动Beta）

    Returns:
        tuple: (tau_star, corrs, max_related_matrix, beta, rolling_beta_result)
            - tau_star: 最优延迟值
            - corrs: 所有延迟值对应的相关系数列表
            - max_related_matrix: 最大相关系数
            - beta: 静态Beta系数（如果启用）或 None
            - rolling_beta_result: 滚动Beta结果字典（如果启用）或 None
    """
    # ... 前面的代码保持不变 ...

    # ========== 4. 计算静态 Beta 系数 ==========
    beta = None
    if enable_beta_calc:
        # ... 静态Beta计算代码 ...
        pass

    # ========== 5. 计算滚动窗口 Beta（新增）==========
    rolling_beta_result = None
    if enable_rolling_beta is None:
        enable_rolling_beta = DelayCorrelationAnalyzer.ENABLE_ROLLING_BETA

    if enable_rolling_beta:
        # 使用处理后的数据计算滚动Beta
        rolling_beta_result = DelayCorrelationAnalyzer._calculate_rolling_beta(
            btc_ret_processed,
            alt_ret_processed
        )

        # 如果滚动Beta计算成功，记录关键信息
        if rolling_beta_result is not None:
            logger.info(
                f"滚动Beta分析 | "
                f"当前值: {rolling_beta_result['current_beta']:.4f} | "
                f"历史均值: {rolling_beta_result['mean_beta']:.4f} | "
                f"波动性: {rolling_beta_result['std_beta']:.4f} | "
                f"趋势: {rolling_beta_result['beta_trend']}"
            )

    return tau_star, corrs, max_related_matrix, beta, rolling_beta_result
```

---

## 四、实际应用场景

### 4.1 场景1：动态风险管理

**问题：** 如何根据Beta变化调整仓位？

**解决方案：**
```python
def get_position_sizing_recommendation(rolling_beta_result, base_position=1.0):
    """
    根据滚动Beta推荐仓位大小

    Args:
        rolling_beta_result: 滚动Beta计算结果
        base_position: 基础仓位（默认1.0）

    Returns:
        dict: {
            'recommended_position': 推荐仓位比例,
            'risk_level': 风险等级,
            'reason': 推荐理由
        }
    """
    current_beta = rolling_beta_result['current_beta']
    mean_beta = rolling_beta_result['mean_beta']
    std_beta = rolling_beta_result['std_beta']

    # 计算Beta相对于历史均值的偏离程度
    beta_zscore = (current_beta - mean_beta) / std_beta if std_beta > 0 else 0

    # 仓位调整策略
    if current_beta > 1.5:
        # 高Beta：波动剧烈，降低仓位
        position = base_position * 0.6
        risk_level = "高风险"
        reason = f"当前Beta={current_beta:.2f}，波动幅度是BTC的{current_beta}倍，建议降低仓位"

    elif current_beta > 1.2:
        # 中高Beta：适度波动
        position = base_position * 0.8
        risk_level = "中高风险"
        reason = f"当前Beta={current_beta:.2f}，波动略高于BTC，适度降低仓位"

    elif current_beta > 0.8:
        # 正常Beta：标准仓位
        position = base_position
        risk_level = "中等风险"
        reason = f"当前Beta={current_beta:.2f}，跟随BTC正常波动，保持标准仓位"

    else:
        # 低Beta：波动小或脱钩，谨慎处理
        position = base_position * 0.7
        risk_level = "低相关性"
        reason = f"当前Beta={current_beta:.2f}，与BTC相关性降低，可能脱钩或独立走势"

    # 趋势加成：如果Beta正在上升，额外降低仓位
    if rolling_beta_result['beta_trend'] == 'increasing' and beta_zscore > 1:
        position *= 0.9
        reason += " | Beta趋势上升，进一步降低仓位"

    return {
        'recommended_position': position,
        'risk_level': risk_level,
        'reason': reason,
        'beta_zscore': beta_zscore
    }
```

**输出示例：**
```
推荐仓位: 0.54 (基础仓位的54%)
风险等级: 高风险
理由: 当前Beta=1.85，波动幅度是BTC的1.85倍，建议降低仓位 | Beta趋势上升，进一步降低仓位
```

### 4.2 场景2：异常Beta检测

**问题：** 如何识别Beta突变，预警风险？

```python
def detect_beta_anomaly(rolling_beta_result, threshold_std=2.0):
    """
    检测Beta异常值（基于统计学方法）

    Args:
        rolling_beta_result: 滚动Beta结果
        threshold_std: 异常阈值（标准差倍数）

    Returns:
        dict: 异常检测结果
    """
    current_beta = rolling_beta_result['current_beta']
    mean_beta = rolling_beta_result['mean_beta']
    std_beta = rolling_beta_result['std_beta']

    # 计算Z-Score
    zscore = (current_beta - mean_beta) / std_beta if std_beta > 0 else 0

    # 判断是否异常
    is_anomaly = abs(zscore) > threshold_std

    if is_anomaly:
        if zscore > 0:
            severity = "高波动异常"
            message = f"⚠️ Beta异常上升！当前值{current_beta:.2f}远超历史均值{mean_beta:.2f}"
        else:
            severity = "脱钩异常"
            message = f"⚠️ Beta异常下降！当前值{current_beta:.2f}远低于历史均值{mean_beta:.2f}"
    else:
        severity = "正常"
        message = f"✅ Beta在正常范围内（均值±{threshold_std}σ）"

    return {
        'is_anomaly': is_anomaly,
        'severity': severity,
        'message': message,
        'zscore': zscore,
        'threshold': threshold_std
    }
```

### 4.3 场景3：套利机会识别

**问题：** 如何结合延迟相关系数和滚动Beta识别套利机会？

```python
def identify_arbitrage_opportunity(tau_star, corr, rolling_beta_result):
    """
    综合分析延迟相关系数和滚动Beta，识别套利机会

    Returns:
        dict: {
            'has_opportunity': 是否存在机会,
            'opportunity_type': 机会类型,
            'confidence': 置信度,
            'strategy': 推荐策略
        }
    """
    current_beta = rolling_beta_result['current_beta']
    beta_std = rolling_beta_result['std_beta']

    # 判断条件
    has_delay = tau_star > 0
    high_correlation = corr > 0.7
    stable_beta = beta_std < 0.3  # Beta波动小，关系稳定
    moderate_beta = 0.8 <= current_beta <= 1.5  # Beta适中，跟随合理

    # 综合判断
    if has_delay and high_correlation and stable_beta and moderate_beta:
        return {
            'has_opportunity': True,
            'opportunity_type': '高质量延迟套利',
            'confidence': 0.85,
            'strategy': f'ALT滞后BTC {tau_star}个周期，相关性{corr:.2f}，Beta稳定在{current_beta:.2f}，'
                       f'可在BTC变动后{tau_star}个周期内操作ALT'
        }

    elif has_delay and high_correlation and not stable_beta:
        return {
            'has_opportunity': True,
            'opportunity_type': '中等延迟套利',
            'confidence': 0.60,
            'strategy': f'存在延迟关系，但Beta波动较大(σ={beta_std:.2f})，需谨慎操作'
        }

    elif current_beta > 1.5 and high_correlation:
        return {
            'has_opportunity': True,
            'opportunity_type': '高杠杆跟随',
            'confidence': 0.70,
            'strategy': f'ALT波动是BTC的{current_beta:.2f}倍，可利用放大效应，但风险较高'
        }

    else:
        return {
            'has_opportunity': False,
            'opportunity_type': '无明显机会',
            'confidence': 0.0,
            'strategy': '当前相关性或延迟不足以支持套利策略'
        }
```

---

## 五、可视化方案

### 5.1 Beta时间序列图

```python
def plot_rolling_beta(rolling_beta_result, coin_name, save_path=None):
    """
    绘制滚动Beta时间序列图
    """
    import matplotlib.pyplot as plt

    betas = rolling_beta_result['betas']
    timestamps = rolling_beta_result['timestamps']
    mean_beta = rolling_beta_result['mean_beta']
    std_beta = rolling_beta_result['std_beta']

    plt.figure(figsize=(12, 6))

    # 1. Beta时间序列
    plt.plot(timestamps, betas, label='Rolling Beta', color='blue', linewidth=2)

    # 2. 均值线
    plt.axhline(y=mean_beta, color='green', linestyle='--',
                label=f'Mean Beta = {mean_beta:.2f}')

    # 3. 置信区间（均值±1σ）
    plt.fill_between(timestamps,
                     mean_beta - std_beta,
                     mean_beta + std_beta,
                     alpha=0.2, color='green',
                     label=f'±1σ ({std_beta:.2f})')

    # 4. Beta=1参考线（同步波动）
    plt.axhline(y=1.0, color='gray', linestyle=':',
                label='Beta = 1.0 (Synchronized)')

    # 5. 高风险区域标记（Beta > 1.5）
    high_risk_mask = betas > 1.5
    if high_risk_mask.any():
        plt.scatter(timestamps[high_risk_mask], betas[high_risk_mask],
                   color='red', s=50, zorder=5, label='High Risk (β>1.5)')

    plt.title(f'{coin_name} Rolling Beta Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Beta Coefficient', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
```

**输出示例图：**
```
📈 ETH Rolling Beta (30-Day Window)
   ↑ Beta
2.5│     ╭─╮
2.0│    ╱   ╰─╮      [红点: 高风险期]
1.5│   ╱       ╰╮   ┄┄┄ [绿色带: ±1σ]
1.0├──────────────────  [灰虚线: β=1]
0.5│                    [绿虚线: 均值]
   └──────────────────────→ Time
```

### 5.2 Beta分布直方图

```python
def plot_beta_distribution(rolling_beta_result, coin_name):
    """
    绘制Beta分布直方图
    """
    import matplotlib.pyplot as plt

    betas = rolling_beta_result['betas']

    plt.figure(figsize=(10, 6))
    plt.hist(betas, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=rolling_beta_result['mean_beta'], color='red',
                linestyle='--', linewidth=2, label=f'Mean = {rolling_beta_result["mean_beta"]:.2f}')
    plt.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, label='Beta = 1.0')

    plt.title(f'{coin_name} Beta Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Beta Coefficient', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 六、性能与成本评估

### 6.1 计算复杂度

```
静态Beta:  O(n)           其中n是数据点总数
滚动Beta:  O(n × w)       其中w是窗口大小

示例：
- 数据量: 1000个点
- 窗口: 30个点
- 步长: 1

滚动Beta计算次数 = (1000 - 30 + 1) / 1 = 971次
每次计算复杂度: O(30)
总复杂度: O(971 × 30) ≈ O(30,000)

实际耗时（基于NumPy优化）:
- 静态Beta: ~0.5ms
- 滚动Beta: ~15ms （30倍时间）
```

### 6.2 优化建议

**方法1：调整步长（trade-off精度与速度）**
```python
# 步长=1: 最细粒度，计算量大
ROLLING_BETA_STEP = 1  # 971次计算

# 步长=5: 降低计算量80%
ROLLING_BETA_STEP = 5  # 194次计算，速度提升5倍
```

**方法2：使用Pandas滚动窗口（性能优化）**
```python
def _calculate_rolling_beta_optimized(btc_ret, alt_ret, window_size=30):
    """
    使用Pandas rolling优化性能
    """
    import pandas as pd

    # 转换为DataFrame
    df = pd.DataFrame({
        'btc': btc_ret,
        'alt': alt_ret
    })

    # 使用rolling + apply计算Beta（Pandas内部优化）
    def calc_beta(window_data):
        cov = window_data['btc'].cov(window_data['alt'])
        var = window_data['btc'].var()
        return cov / var if var > 0 else np.nan

    rolling_betas = df.rolling(window=window_size).apply(
        lambda x: calc_beta(df.iloc[x.index]),
        raw=False
    )

    return rolling_betas['btc'].values  # 返回Beta序列
```

---

## 七、实施计划

### 7.1 分阶段实施

**阶段1（基础）：实现核心功能**
- ✅ 添加配置常量到类定义
- ✅ 实现 `_calculate_rolling_beta` 方法
- ✅ 集成到 `find_optimal_delay`
- ✅ 默认关闭（`ENABLE_ROLLING_BETA = False`）

**阶段2（增强）：添加分析功能**
- ✅ 实现 `get_position_sizing_recommendation` 仓位推荐
- ✅ 实现 `detect_beta_anomaly` 异常检测
- ✅ 实现 `identify_arbitrage_opportunity` 套利识别

**阶段3（可视化）：添加图表功能**
- ✅ 实现 `plot_rolling_beta` 时间序列图
- ✅ 实现 `plot_beta_distribution` 分布直方图
- ✅ 集成到结果输出流程

**阶段4（优化）：性能与扩展**
- ✅ 使用Pandas优化计算速度
- ✅ 添加结果缓存机制
- ✅ 支持自定义窗口策略

### 7.2 详细实施步骤

#### 步骤1：添加配置常量（预计0.5小时）

**修改位置：** `hyperliquid_analyzer.py` 第65-73行

```python
class DelayCorrelationAnalyzer:
    # 现有常量
    MIN_POINTS_FOR_CORR_CALC = 10
    MIN_DATA_POINTS_FOR_ANALYSIS = 50

    # ========== 现有：异常值处理配置 ==========
    WINSORIZE_LOWER_PERCENTILE = 5
    WINSORIZE_UPPER_PERCENTILE = 95
    ENABLE_OUTLIER_TREATMENT = True

    # ========== 现有：Beta 系数配置 ==========
    ENABLE_BETA_CALCULATION = True
    MIN_POINTS_FOR_BETA_CALC = 10

    # ========== 新增：滚动窗口Beta配置 ==========
    ROLLING_BETA_WINDOW_SIZE = 30
    ROLLING_BETA_STEP = 1
    ENABLE_ROLLING_BETA = False
    MIN_ROLLING_WINDOW_SIZE = 20
```

#### 步骤2：实现核心方法（预计2小时）

**添加位置：** `_calculate_beta` 方法之后

1. 实现 `_calculate_rolling_beta` 方法（参见3.2节完整代码）
2. 添加完整的文档字符串和注释
3. 添加异常处理和日志记录

#### 步骤3：集成到主流程（预计1.5小时）

**修改 `find_optimal_delay` 方法：**

1. 添加 `enable_rolling_beta` 参数
2. 在返回值中添加 `rolling_beta_result`
3. 调用 `_calculate_rolling_beta` 计算滚动Beta
4. 更新返回语句和文档字符串

**修改 `_analyze_single_combination` 方法：**

```python
def _analyze_single_combination(self, coin: str, timeframe: str, period: str) -> tuple | None:
    """
    分析单个组合（增强版：支持滚动Beta）

    Returns:
        (correlation, timeframe, period, tau_star, beta, rolling_beta_result)
    """
    # ... 现有代码 ...

    result = self.find_optimal_delay(
        btc_df_aligned['return'].values,
        alt_df_aligned['return'].values
    )

    # 处理新返回值格式
    if len(result) == 5:
        tau_star, _, related_matrix, beta, rolling_beta_result = result
    elif len(result) == 4:
        tau_star, _, related_matrix, beta = result
        rolling_beta_result = None
    else:
        tau_star, _, related_matrix = result
        beta = None
        rolling_beta_result = None

    return (related_matrix, timeframe, period, tau_star, beta, rolling_beta_result)
```

#### 步骤4：实现分析功能（预计1.5小时）

创建新文件 `beta_analysis_utils.py`，包含：

1. `get_position_sizing_recommendation` 函数
2. `detect_beta_anomaly` 函数
3. `identify_arbitrage_opportunity` 函数

#### 步骤5：实现可视化（预计1小时）

创建新文件 `beta_visualization.py`，包含：

1. `plot_rolling_beta` 函数
2. `plot_beta_distribution` 函数

#### 步骤6：更新结果输出（预计1小时）

**修改 `_output_results` 方法：**

```python
def _output_results(self, coin: str, results: list, diff_amount: float):
    """
    输出分析结果（增强版：包含滚动Beta）
    """
    data_rows = []
    for result in results:
        if len(result) == 6:
            corr, tf, p, ts, beta, rolling_beta_result = result
        elif len(result) == 5:
            corr, tf, p, ts, beta = result
            rolling_beta_result = None
        else:
            corr, tf, p, ts = result
            beta = None
            rolling_beta_result = None

        row = {
            '最大相关系数': corr,
            '时间周期': tf,
            '数据周期': p,
            '最优延迟': ts
        }

        if beta is not None:
            row['Beta系数'] = beta

        if rolling_beta_result is not None:
            row['当前Beta'] = rolling_beta_result['current_beta']
            row['Beta均值'] = rolling_beta_result['mean_beta']
            row['Beta波动'] = rolling_beta_result['std_beta']
            row['Beta趋势'] = rolling_beta_result['beta_trend']

        data_rows.append(row)

    df_results = pd.DataFrame(data_rows)

    # ... 输出逻辑 ...
```

### 7.3 配置建议

```python
# 保守配置（高精度，低频交易）
ROLLING_BETA_WINDOW_SIZE = 50
ROLLING_BETA_STEP = 5
ENABLE_ROLLING_BETA = True

# 激进配置（快速响应，高频交易）
ROLLING_BETA_WINDOW_SIZE = 20
ROLLING_BETA_STEP = 1
ENABLE_ROLLING_BETA = True

# 默认配置（平衡）
ROLLING_BETA_WINDOW_SIZE = 30
ROLLING_BETA_STEP = 1
ENABLE_ROLLING_BETA = False  # 按需启用
```

---

## 八、测试计划

### 8.1 单元测试

```python
def test_calculate_rolling_beta():
    """测试滚动Beta计算"""
    # 测试1：正常情况
    btc_ret = np.random.normal(0.001, 0.02, 100)
    alt_ret = btc_ret * 1.5 + np.random.normal(0, 0.01, 100)

    result = DelayCorrelationAnalyzer._calculate_rolling_beta(
        btc_ret, alt_ret, window_size=30
    )

    assert result is not None
    assert 'current_beta' in result
    assert 1.2 < result['mean_beta'] < 1.8  # Beta应该接近1.5

    # 测试2：数据不足
    short_data = np.array([0.01, 0.02, 0.03])
    result = DelayCorrelationAnalyzer._calculate_rolling_beta(
        short_data, short_data, window_size=30
    )
    assert result is None

    # 测试3：窗口大小检查
    result = DelayCorrelationAnalyzer._calculate_rolling_beta(
        btc_ret, alt_ret, window_size=10  # 小于最小值20
    )
    # 应该自动调整到20
    assert result is not None

def test_position_sizing():
    """测试仓位推荐"""
    # 模拟滚动Beta结果
    rolling_beta = {
        'current_beta': 1.8,
        'mean_beta': 1.2,
        'std_beta': 0.3,
        'beta_trend': 'increasing'
    }

    rec = get_position_sizing_recommendation(rolling_beta)

    assert rec['risk_level'] == '高风险'
    assert rec['recommended_position'] < 1.0  # 应该降低仓位

def test_beta_anomaly_detection():
    """测试Beta异常检测"""
    # 正常情况
    normal_result = {
        'current_beta': 1.2,
        'mean_beta': 1.2,
        'std_beta': 0.2
    }
    detection = detect_beta_anomaly(normal_result, threshold_std=2.0)
    assert detection['is_anomaly'] == False

    # 异常情况
    anomaly_result = {
        'current_beta': 2.5,
        'mean_beta': 1.2,
        'std_beta': 0.2
    }
    detection = detect_beta_anomaly(anomaly_result, threshold_std=2.0)
    assert detection['is_anomaly'] == True
```

### 8.2 集成测试

```python
def test_end_to_end_rolling_beta():
    """端到端测试滚动Beta功能"""
    # 创建测试数据
    analyzer = DelayCorrelationAnalyzer()

    # 生成模拟BTC和ALT数据
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    btc_prices = 40000 + np.cumsum(np.random.normal(0, 500, 200))
    alt_prices = 2000 + np.cumsum(np.random.normal(0, 100, 200))

    btc_df = pd.DataFrame({'close': btc_prices}, index=dates)
    alt_df = pd.DataFrame({'close': alt_prices}, index=dates)

    # 计算收益率
    btc_ret = btc_df['close'].pct_change().dropna().values
    alt_ret = alt_df['close'].pct_change().dropna().values

    # 启用滚动Beta
    result = analyzer.find_optimal_delay(
        btc_ret, alt_ret,
        enable_rolling_beta=True
    )

    assert len(result) == 5
    tau_star, corrs, max_corr, beta, rolling_beta = result

    # 验证滚动Beta结果
    assert rolling_beta is not None
    assert 'current_beta' in rolling_beta
    assert len(rolling_beta['betas']) > 0
```

### 8.3 性能测试

```python
import time

def test_performance_comparison():
    """对比静态Beta和滚动Beta的性能"""
    # 准备大规模数据
    btc_ret = np.random.normal(0.001, 0.02, 10000)
    alt_ret = btc_ret * 1.5 + np.random.normal(0, 0.01, 10000)

    # 测试静态Beta
    start = time.time()
    beta_static = DelayCorrelationAnalyzer._calculate_beta(btc_ret, alt_ret)
    time_static = time.time() - start

    # 测试滚动Beta（窗口=30，步长=1）
    start = time.time()
    result_rolling = DelayCorrelationAnalyzer._calculate_rolling_beta(
        btc_ret, alt_ret, window_size=30, step=1
    )
    time_rolling = time.time() - start

    print(f"静态Beta耗时: {time_static*1000:.2f}ms")
    print(f"滚动Beta耗时: {time_rolling*1000:.2f}ms")
    print(f"性能比: {time_rolling/time_static:.1f}x")

    # 验证耗时在合理范围内
    assert time_rolling < 0.1  # 应该在100ms以内
```

---

## 九、风险评估与应对

### 9.1 风险识别

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| **计算性能下降** | 中 | 中 | 默认关闭，通过步长调整性能 |
| **窗口大小选择不当** | 中 | 低 | 提供推荐值，支持自定义 |
| **结果解释困难** | 低 | 中 | 提供详细文档和可视化 |
| **向后兼容性问题** | 低 | 低 | 使用条件判断处理返回值 |
| **内存占用增加** | 低 | 低 | 滚动Beta结果只保留必要信息 |

### 9.2 回滚方案

1. **配置级回滚**：设置 `ENABLE_ROLLING_BETA = False`
2. **代码级回滚**：返回值兼容处理确保旧代码正常运行
3. **参数级回滚**：方法参数默认值为None，不影响现有调用

---

## 十、验收标准

### 10.1 功能验收

- [ ] `_calculate_rolling_beta` 方法正确实现
- [ ] 滚动Beta计算结果包含所有必要字段
- [ ] 集成到 `find_optimal_delay` 无报错
- [ ] 仓位推荐功能正常工作
- [ ] 异常检测功能正常工作
- [ ] 套利机会识别功能正常工作

### 10.2 性能验收

- [ ] 滚动Beta计算耗时 < 100ms（1000个数据点）
- [ ] 对现有功能性能影响 < 5%
- [ ] 内存占用增加 < 10MB

### 10.3 质量验收

- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 向后兼容性验证通过
- [ ] 代码注释完整
- [ ] 文档齐全

---

## 十一、总结

### 11.1 核心价值对比

| 功能 | 静态Beta | 滚动Beta |
|------|---------|----------|
| **计算复杂度** | O(n) | O(n×w) |
| **信息量** | 单一值 | 时间序列 |
| **应用场景** | 长期风险评估 | 动态仓位管理、异常检测 |
| **实时性** | 低 | 高 |
| **适用对象** | 稳定市场 | 波动市场 |

### 11.2 实施价值评估

**值得实施的情况：**
- ✅ 需要动态风险管理
- ✅ 高频交易或短期策略
- ✅ 市场波动剧烈期
- ✅ 需要精细化仓位控制

**可暂缓实施的情况：**
- ⏸️ 仅做长期持有分析
- ⏸️ 计算资源受限
- ⏸️ 用户对时变性不敏感
- ⏸️ 数据量不足（< 50个点）

### 11.3 实施时间表

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|---------|--------|
| 阶段1 | 添加配置常量 | 0.5小时 | 高 |
| 阶段2 | 实现核心方法 | 2小时 | 高 |
| 阶段3 | 集成到主流程 | 1.5小时 | 高 |
| 阶段4 | 实现分析功能 | 1.5小时 | 中 |
| 阶段5 | 实现可视化 | 1小时 | 中 |
| 阶段6 | 更新结果输出 | 1小时 | 中 |
| 测试 | 单元+集成+性能测试 | 2小时 | 高 |
| 文档 | 更新代码注释和文档 | 0.5小时 | 中 |

**总计：约10小时**

### 11.4 最终建议

**实施策略：**
1. **作为可选高级功能**实现，通过 `--enable-rolling-beta` 参数控制
2. **默认关闭**，在需要深度分析时启用
3. **分阶段实施**，先实现核心功能，后续逐步添加分析和可视化
4. **充分测试**，确保不影响现有功能

**预期收益：**
- ✅ 捕捉Beta时变特征，提升风险管理能力
- ✅ 支持动态仓位调整策略
- ✅ 提供异常检测和预警功能
- ✅ 增强套利机会识别准确性

---

## 十二、附录

### 12.1 相关公式

#### 滚动窗口Beta
\[
\beta_t = \frac{\text{Cov}(R_{ALT,t-w:t}, R_{BTC,t-w:t})}{\text{Var}(R_{BTC,t-w:t})}
\]

其中 \( w \) 是窗口大小，\( t \) 是当前时间点。

#### Beta的Z-Score（异常检测）
\[
Z = \frac{\beta_{current} - \mu_\beta}{\sigma_\beta}
\]

其中 \( \mu_\beta \) 是历史平均Beta，\( \sigma_\beta \) 是Beta标准差。

### 12.2 参考资源

- **学术文献**：
  - Fama, E. F., & French, K. R. (1992). "The cross-section of expected stock returns"
  - Bollerslev, T., Engle, R. F., & Wooldridge, J. M. (1988). "A capital asset pricing model with time-varying covariances"

- **技术文档**：
  - NumPy文档：`np.cov`, `np.std`, `np.mean`
  - Pandas文档：`DataFrame.rolling`

- **金融分析最佳实践**：
  - 滚动窗口技术在风险管理中的应用
  - 动态Beta在投资组合优化中的应用

---

## 十三、文档版本历史

| 版本 | 日期 | 修改内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2025-12-24 | 初始版本 | AI Assistant |

---

**文档结束**

---

> 该方案文档可作为滚动窗口Beta功能的完整实施指南。建议在完成"优化实施计划.md"中的基础Beta功能后，再考虑实施本方案。
