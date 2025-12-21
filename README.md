# 相关系数异常告警系统

- 一个用于分析山寨币与BTC皮尔逊相关系数的工具，
    - 通过识别短期高相关但长期低相关的异常模式，发现存在时间差套利空间的币种。
    - 通过识别短期低相关但长期高相关的币种，找到短期存在滞后性但长期跟随BTC的代币
    
- 基于项目 https://github.com/zhajingwen/calculate_fake_TE/blob/main/corrcoef_abnormal.py  开发

## 项目简介

本项目实现了虚假传递熵（Spurious Transfer Entropy）分析算法，用于检测加密货币市场中与BTC存在异常相关性的币种。通过计算不同时间周期和延迟下的相关系数，识别出两类异常模式：

1. **模式1（Binance版）**：短期（1分钟K线）最大相关系数 > 0.6，长期（60天）最大相关系数 < 0.2，差值 > 0.5
   - 表示短期存在很大滞后性，但长期跟随性弱，存在时间差套利空间

2. **模式2（Hyperliquid版）**：长期最大相关系数 > 0.6，短期最小相关系数 < 0.3，差值 > 0.5
   - 表示短期存在明显滞后性，但长期与BTC有较强的跟随性，存在时间差套利机会
   - 额外条件：当1天周期的最优延迟（tau_star）> 0时也会触发告警

## 核心功能

- 📊 **多周期分析**：支持1分钟、5分钟K线，以及1天、7天、30天、60天的数据周期
- 🔍 **最优延迟计算**：通过寻找最优延迟（tau*）识别币种与BTC的滞后关系
- 📈 **异常模式识别**：自动识别存在套利机会的异常币种
- 🔔 **飞书告警**：检测到异常币种时自动发送告警到飞书群
- ⏰ **定时调度**：支持定时任务调度，可配置执行时间
- 🔄 **自动重试**：网络请求失败时自动重试，提高稳定性
- 🏦 **多交易所支持**：支持 Binance、Hyperliquid 等多个交易所

## 技术架构

### 核心算法

1. **数据下载**：从交易所API下载BTC和山寨币的历史K线数据（OHLCV）
2. **收益率计算**：计算相邻K线的价格变化百分比
3. **延迟分析**：计算不同延迟（lag）下BTC和山寨币收益率的相关系数
4. **最优延迟**：找出使相关系数最大的延迟值（tau*）
5. **异常检测**：根据短期和长期相关系数判断是否为异常模式

### 项目结构

```
related_corrcoef_abnormal_alert/
├── corrcoef_abnormal_alert_to_lark.py          # 主程序（Binance，带飞书告警）
├── corrcoef_abnormal_alert_to_lark_except_outstanding.py  # 排除持仓版本（Binance）
├── corrcoef_abnormal_print.py                  # 仅打印结果版本
├── hyperliquid.py                              # Hyperliquid永续合约分析
├── pyproject.toml                              # 项目配置和依赖
├── README.md                                   # 项目文档
└── utils/                                      # 工具模块
    ├── __init__.py
    ├── config.py                               # 配置文件（环境变量）
    ├── lark_bot.py                             # 飞书机器人消息发送
    ├── scheduler.py                            # 定时调度装饰器
    └── spider_failed_alert.py                  # 爬虫失败告警装饰器
```

## 环境要求

- Python >= 3.12
- 支持的交易所：Binance（默认）、Hyperliquid、KuCoin 或ccxt库支持的其他交易所

## 安装

1. 克隆项目：
```bash
git clone <repository-url>
cd related_corrcoef_abnormal_alert
```

2. 安装依赖（推荐使用uv）：
```bash
uv sync
```

或使用pip：
```bash
pip install -r requirements.txt
```

## 配置

### 环境变量

在运行前需要配置以下环境变量（可选）：

- `LARKBOT_ID`: 飞书机器人Webhook ID（用于告警）
- `ENV`: 运行环境（`local` 或 `prod`，默认为 `local`）
- `REDIS_HOST`: Redis主机地址（默认为 `127.0.0.1`）
- `REDIS_PASSWORD`: Redis密码（如果使用Redis）

### 飞书机器人配置

1. 在飞书群中创建自定义机器人
2. 获取Webhook地址中的bot_id
3. 设置环境变量 `LARKBOT_ID` 或在代码中配置

## 使用方法

### 基本使用

#### Binance 交易所分析（默认）

运行主程序（带飞书告警）：
```bash
python corrcoef_abnormal_alert_to_lark.py
```

运行排除持仓版本：
```bash
python corrcoef_abnormal_alert_to_lark_except_outstanding.py
```

或运行仅打印版本：
```bash
python corrcoef_abnormal_print.py
```

#### Hyperliquid 永续合约分析

```bash
python hyperliquid.py
```

### 自定义配置

```python
from corrcoef_abnormal_alert_to_lark import SpuriousTEAnalyzer

# 创建分析器实例
analyzer = SpuriousTEAnalyzer(
    exchange_name="binance",           # 交易所名称（默认binance）
    timeout=30000,                     # 请求超时时间（毫秒）
    default_timeframes=["1m", "5m"],   # 时间周期列表
    default_periods=["1d", "7d", "30d", "60d"]  # 数据周期列表
)

# 分析所有USDT交易对
analyzer.run(quote_currency="USDT")
```

### 定时调度

使用 `scheduler.py` 中的装饰器实现定时任务：

```python
from utils.scheduler import scheduled_task

@scheduled_task(start_time="09:00", weekdays=[0, 1, 2, 3, 4])  # 工作日9点执行
def my_analysis():
    analyzer = SpuriousTEAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    my_analysis()
```

调度方式：
- **周几的几点执行**：提供 `start_time` 和 `weekdays` 参数
- **每天的几点执行**：只提供 `start_time` 参数
- **每隔N秒执行一次**：只提供 `duration` 参数

## 依赖说明

主要依赖包：

- `ccxt`: 加密货币交易所API统一接口
- `numpy`: 数值计算和相关系数计算
- `pandas`: 数据处理和DataFrame操作
- `retry`: 网络请求失败时自动重试
- `matplotlib`, `seaborn`: 数据可视化（可选）

完整依赖列表请查看 `pyproject.toml`。

## 工作原理

### 1. 数据下载

- 从交易所API下载指定交易对的OHLCV历史数据
- 支持分批次下载（每次最多1500条K线）
- 自动计算收益率和美元成交量
- 使用BTC数据缓存避免重复下载

### 2. 延迟分析

对于每个时间周期和数据周期的组合：
- 计算BTC和山寨币的收益率序列
- 遍历不同的延迟值（0到48个时间单位）
- 计算每个延迟下的皮尔逊相关系数
- 找出使相关系数最大的最优延迟（tau*）

### 3. 异常检测

分析所有组合的相关系数结果：
- 按最大相关系数排序
- 比较短期和长期的相关系数
- 判断是否符合异常模式条件
- 如果符合，输出详细分析结果并发送告警

## 输出示例

检测到异常币种时，会输出如下格式的分析结果：

```
============================================================
ETH/USDT相关系数分析结果
  最大相关系数  时间周期 数据周期  最优延迟
        0.65       1m     1d        12
        0.45       5m     7d         8
        0.12       1m    30d         5
        0.03       5m    60d         2

 diff_amount: 0.62
============================================================
```

## 版本差异说明

| 版本 | 交易所 | 交易对类型 | 异常模式 |
|------|--------|------------|----------|
| `corrcoef_abnormal_alert_to_lark.py` | Binance | USDT现货 | 短期高相关 + 长期低相关 |
| `corrcoef_abnormal_alert_to_lark_except_outstanding.py` | Binance | USDT现货 | 同上，排除持仓 |
| `corrcoef_abnormal_print.py` | KuCoin | USDT现货 | 短期高相关 + 长期低相关 |
| `hyperliquid.py` | Hyperliquid | USDC永续 | 短期低相关 + 长期高相关 |

## 注意事项

1. **API限制**：交易所API通常有速率限制，程序在每次分析后等待1秒以避免触发限流
2. **数据量**：分析所有币种可能需要较长时间，取决于交易所的交易对数量
3. **网络稳定性**：程序内置重试机制，但网络不稳定时可能需要多次重试
4. **计算资源**：处理大量数据时需要足够的内存和计算资源
5. **合约过滤**：程序会自动过滤合约交易对（如 `/USDT:USDT`），只分析现货交易对

## 开发说明

### 本地开发

设置 `ENV=local` 环境变量，定时调度装饰器会直接执行任务而不等待调度时间。

### 错误处理

- 使用 `@retry` 装饰器自动重试失败的API请求
- 使用 `ErrorMonitor` 装饰器捕获异常并发送告警
- 数据为空或格式错误时会跳过并记录警告
- 单个币种分析失败不会影响其他币种的处理

## 许可证

[添加许可证信息]

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过Issue联系。
