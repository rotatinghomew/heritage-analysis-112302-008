"""工具内的异常类型。"""

from __future__ import annotations


class CraftReviewError(Exception):
    """所有可预期错误的基类，便于命令行统一转成退出码。"""


class InputError(CraftReviewError):
    """输入文件缺失、字段错误或记录自相矛盾。"""


class PolicyError(InputError):
    """策略表违反标注手册或取值非法。"""


class IsolationError(CraftReviewError):
    """片源家族隔离被破坏，无法产生可信评估。"""
