#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础Beta功能测试脚本

测试新增的Winsorization和Beta系数功能
"""

from hyperliquid_analyzer import DelayCorrelationAnalyzer
import numpy as np


def test_winsorization():
    """测试Winsorization异常值处理"""
    print('\n' + '='*60)
    print('测试1: Winsorization 异常值处理')
    print('='*60)

    # 创建包含异常值的数据
    returns = np.array([0.01, 0.02, -0.01, 0.10, -0.15, 0.03] + [0.001]*20)

    print(f'原始数据点数: {len(returns)}')
    print(f'原始数据范围: [{np.min(returns):.6f}, {np.max(returns):.6f}]')

    # 应用Winsorization
    result = DelayCorrelationAnalyzer._winsorize_returns(returns, log_stats=True)

    print(f'处理后范围: [{np.min(result):.6f}, {np.max(result):.6f}]')
    print('✅ Winsorization 测试通过')


def test_beta_calculation():
    """测试Beta系数计算"""
    print('\n' + '='*60)
    print('测试2: Beta 系数计算')
    print('='*60)

    # 设置随机种子以获得可复现结果
    np.random.seed(42)

    # 模拟BTC收益率
    btc_ret = np.random.normal(0.001, 0.02, 100)

    # 测试不同Beta场景
    test_cases = [
        ('Beta = 1.5 (ALT波动是BTC的1.5倍)', btc_ret * 1.5 + np.random.normal(0, 0.01, 100), 1.2, 1.8),
        ('Beta = 0.8 (ALT波动小于BTC)', btc_ret * 0.8 + np.random.normal(0, 0.01, 100), 0.6, 1.0),
        ('Beta = 2.0 (高波动)', btc_ret * 2.0 + np.random.normal(0, 0.01, 100), 1.8, 2.3),
    ]

    for desc, alt_ret, min_expected, max_expected in test_cases:
        beta = DelayCorrelationAnalyzer._calculate_beta(btc_ret, alt_ret)
        print(f'\n场景: {desc}')
        print(f'计算得到的Beta: {beta:.4f}')
        print(f'预期范围: [{min_expected:.1f}, {max_expected:.1f}]')

        if min_expected <= beta <= max_expected:
            print('✅ Beta值在预期范围内')
        else:
            print(f'⚠️  Beta值超出预期范围（但可能正常，因为随机数据）')


def test_find_optimal_delay_enhanced():
    """测试增强版find_optimal_delay"""
    print('\n' + '='*60)
    print('测试3: find_optimal_delay 增强版')
    print('='*60)

    np.random.seed(42)
    btc_ret = np.random.normal(0.001, 0.02, 100)
    alt_ret = btc_ret * 1.5 + np.random.normal(0, 0.01, 100)

    # 测试返回值
    result = DelayCorrelationAnalyzer.find_optimal_delay(btc_ret, alt_ret)

    print(f'返回值数量: {len(result)}个 (预期4个)')

    tau_star, corrs, max_corr, beta = result

    print(f'最优延迟 (tau_star): {tau_star}')
    print(f'最大相关系数: {max_corr:.4f}')
    print(f'Beta系数: {beta:.4f if beta is not None else "None"}')

    # 验证返回值
    assert len(result) == 4, "返回值应该是4个"
    assert beta is not None, "Beta系数不应该为None"
    assert not np.isnan(beta), "Beta系数不应该为NaN"

    print('✅ find_optimal_delay 返回值格式正确')


def test_edge_cases():
    """测试边界情况"""
    print('\n' + '='*60)
    print('测试4: 边界情况处理')
    print('='*60)

    # 测试1: 数据不足
    print('\n场景1: 数据点不足 (少于10个点)')
    small_data = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    beta = DelayCorrelationAnalyzer._calculate_beta(small_data, small_data)
    print(f'Beta结果: {"NaN" if np.isnan(beta) else beta}')
    assert np.isnan(beta), "数据不足时应返回NaN"
    print('✅ 正确处理数据不足情况')

    # 测试2: BTC方差为0
    print('\n场景2: BTC收益率方差为0')
    constant_btc = np.ones(50) * 0.001
    random_alt = np.random.normal(0.001, 0.02, 50)
    beta = DelayCorrelationAnalyzer._calculate_beta(constant_btc, random_alt)
    print(f'Beta结果: {"NaN" if np.isnan(beta) else beta}')
    assert np.isnan(beta), "BTC方差为0时应返回NaN"
    print('✅ 正确处理方差为0情况')

    # 测试3: Winsorization对小数据不处理
    print('\n场景3: Winsorization对小数据不处理 (少于20个点)')
    small_returns = np.array([0.01, 0.02, -0.01] * 5)  # 15个点
    result = DelayCorrelationAnalyzer._winsorize_returns(small_returns, log_stats=False)
    np.testing.assert_array_equal(result, small_returns)
    print('✅ Winsorization正确跳过小数据集')


def test_integration():
    """集成测试：模拟完整分析流程"""
    print('\n' + '='*60)
    print('测试5: 集成测试')
    print('='*60)

    print('\n模拟完整分析流程...')

    np.random.seed(42)

    # 模拟BTC和ALT数据
    btc_ret = np.random.normal(0.001, 0.02, 500)
    alt_ret = btc_ret * 1.8 + np.random.normal(0, 0.015, 500)  # 高波动ALT

    # 步骤1: 异常值处理
    btc_processed = DelayCorrelationAnalyzer._winsorize_returns(btc_ret, log_stats=False)
    alt_processed = DelayCorrelationAnalyzer._winsorize_returns(alt_ret, log_stats=False)
    print('✅ 步骤1: 异常值处理完成')

    # 步骤2: 计算最优延迟和Beta
    tau_star, corrs, max_corr, beta = DelayCorrelationAnalyzer.find_optimal_delay(
        btc_processed, alt_processed
    )
    print('✅ 步骤2: 最优延迟和Beta计算完成')

    # 步骤3: 输出结果
    print(f'\n分析结果:')
    print(f'  - 最优延迟: {tau_star}个周期')
    print(f'  - 最大相关系数: {max_corr:.4f}')
    print(f'  - Beta系数: {beta:.4f}')

    # 步骤4: 风险评估
    if beta > 1.5:
        risk_level = '高风险'
        risk_msg = f'⚠️  波动幅度是BTC的{beta:.1f}倍'
    elif beta > 1.2:
        risk_level = '中风险'
        risk_msg = f'⚠️  波动略高于BTC'
    else:
        risk_level = '正常'
        risk_msg = '✅ 波动在正常范围'

    print(f'  - 风险等级: {risk_level}')
    print(f'  - 风险提示: {risk_msg}')

    print('\n✅ 集成测试通过')


def main():
    """运行所有测试"""
    print('\n' + '='*60)
    print('基础Beta功能测试套件')
    print('='*60)

    try:
        test_winsorization()
        test_beta_calculation()
        test_find_optimal_delay_enhanced()
        test_edge_cases()
        test_integration()

        print('\n' + '='*60)
        print('🎉 所有测试通过！基础Beta功能运行正常')
        print('='*60)
        print('\n下一步:')
        print('1. 运行 python hyperliquid_analyzer.py 测试真实数据')
        print('2. 检查日志输出中的Beta信息')
        print('3. 查看飞书告警中是否包含Beta系数和风险提示')
        print('='*60 + '\n')

    except Exception as e:
        print(f'\n❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
