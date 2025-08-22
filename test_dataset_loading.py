#!/usr/bin/env python3
"""
数据集加载测试脚本
用于诊断 lm_eval 和 datasets 库的兼容性问题
"""

import sys
import traceback
from typing import Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_imports() -> bool:
    """测试基本库的导入"""
    logger.info("=== 测试基本库导入 ===")
    try:
        import lm_eval
        # 使用 getattr 安全地获取版本，如果不存在则显示为'未知'
        lm_eval_version = getattr(lm_eval, '__version__', '未知')
        logger.info(f"✓ lm_eval 导入成功，版本: {lm_eval_version}")
        
        # 尝试使用 pkg_resources 获取版本（更可靠的方法）
        try:
            import pkg_resources
            lm_eval_pkg_version = pkg_resources.get_distribution("lm-eval").version
            logger.info(f"✓ lm_eval 包版本: {lm_eval_pkg_version}")
        except Exception:
            logger.info("✓ 无法通过 pkg_resources 获取 lm_eval 版本")
        
        import datasets
        logger.info(f"✓ datasets 版本: {datasets.__version__}")
        
        import transformers
        logger.info(f"✓ transformers 版本: {transformers.__version__}")
        
        import torch
        logger.info(f"✓ torch 版本: {torch.__version__}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_evaluator_import() -> Optional[Any]:
    """测试 lm_eval evaluator 的导入和初始化"""
    logger.info("=== 测试 evaluator 导入 ===")
    try:
        from lm_eval import evaluator
        logger.info("✓ evaluator 导入成功")
        return evaluator
    except Exception as e:
        logger.error(f"✗ evaluator 导入失败: {e}")
        traceback.print_exc()
        return None

def test_simple_evaluate_function(evaluator) -> bool:
    """测试 simple_evaluate 函数是否存在且可调用"""
    logger.info("=== 测试 simple_evaluate 函数 ===")
    try:
        if hasattr(evaluator, 'simple_evaluate'):
            func = getattr(evaluator, 'simple_evaluate')
            if callable(func):
                logger.info("✓ simple_evaluate 函数存在且可调用")
                return True
            else:
                logger.error("✗ simple_evaluate 存在但不可调用")
                logger.error(f"simple_evaluate 类型: {type(func)}")
                logger.error(f"simple_evaluate 值: {func}")
                return False
        else:
            logger.error("✗ simple_evaluate 函数不存在")
            logger.info(f"evaluator 可用属性: {dir(evaluator)}")
            return False
    except Exception as e:
        logger.error(f"✗ 检查 simple_evaluate 时出错: {e}")
        traceback.print_exc()
        return False

def test_dataset_loading() -> bool:
    """测试数据集加载"""
    logger.info("=== 测试数据集加载 ===")
    try:
        from datasets import load_dataset
        
        # 测试加载一个简单的数据集
        logger.info("尝试加载 boolq 数据集...")
        dataset = load_dataset('boolq', split='validation[:10]')
        logger.info(f"✓ boolq 数据集加载成功，样本数: {len(dataset)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 数据集加载失败: {e}")
        traceback.print_exc()
        return False

def test_minimal_evaluation(evaluator) -> bool:
    """测试最小化的模型评估"""
    logger.info("=== 测试最小化评估 ===")
    try:
        # 使用一个非常小的测试
        result = evaluator.simple_evaluate(
            model='hf',
            tasks=['boolq'],
            model_args="pretrained=gpt2,dtype=bfloat16,device=cpu",
            num_fewshot=0,
            limit=1  # 只测试1个样本
        )
        logger.info("✓ 最小化评估成功")
        logger.info(f"结果类型: {type(result)}")
        if isinstance(result, dict) and 'results' in result:
            logger.info("✓ 结果包含 'results' 键")
        return True
    except Exception as e:
        logger.error(f"✗ 最小化评估失败: {e}")
        traceback.print_exc()
        return False

def test_specific_tasks() -> bool:
    """测试特定任务的加载"""
    logger.info("=== 测试特定任务加载 ===")
    try:
        from lm_eval.tasks import get_task_dict
        
        # 测试获取任务列表
        tasks = ['piqa', 'boolq', 'race', 'openbookqa', 'sciq', 'lambada_openai']
        for task in tasks:
            try:
                task_dict = get_task_dict([task])
                logger.info(f"✓ 任务 {task} 加载成功")
            except Exception as e:
                logger.error(f"✗ 任务 {task} 加载失败: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 任务加载测试失败: {e}")
        traceback.print_tb()
        return False

def clear_cache() -> None:
    """清理缓存"""
    logger.info("=== 清理缓存 ===")
    try:
        import datasets
        # 清理 datasets 缓存
        datasets.disable_caching()
        logger.info("✓ 禁用 datasets 缓存")
        
    except Exception as e:
        logger.error(f"清理缓存时出错: {e}")

def main():
    """主测试函数"""
    logger.info("开始数据集加载诊断测试")
    logger.info("=" * 50)

    
    # 测试基本导入
    # if not test_basic_imports():
    #     logger.error("基本导入失败，退出测试")
    #     sys.exit(1)
    
    # 测试 evaluator 导入
    evaluator = test_evaluator_import()
    if evaluator is None:
        logger.error("evaluator 导入失败，退出测试")
        sys.exit(1)
    
    # 测试 simple_evaluate 函数
    if not test_simple_evaluate_function(evaluator):
        logger.error("simple_evaluate 函数检查失败")
        # 尝试其他可能的函数名
        logger.info("尝试查找其他评估函数...")
        for attr in dir(evaluator):
            if 'eval' in attr.lower():
                logger.info(f"发现可能的评估函数: {attr}")
    
    # 测试数据集加载
    test_dataset_loading()
    
    # 测试特定任务
    test_specific_tasks()
    
    # 测试最小化评估（如果 simple_evaluate 可用）
    if hasattr(evaluator, 'simple_evaluate') and callable(evaluator.simple_evaluate):
        test_minimal_evaluation(evaluator)
    
    logger.info("=" * 50)
    logger.info("诊断测试完成")
    
    # 提供修复建议
    logger.info("\n=== 修复建议 ===")
    logger.info("如果遇到 'NoneType' object is not callable 错误，请尝试:")
    logger.info("1. 降级 datasets: pip install datasets==2.14.0")
    logger.info("2. 升级 lm_eval: pip install lm_eval[all] --upgrade")
    logger.info("4. 重新安装: pip uninstall datasets lm_eval && pip install datasets==2.14.0 lm_eval[all]")

if __name__ == "__main__":
    main()