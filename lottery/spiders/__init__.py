from .base import BaseProvider
from .cwl import CWLProvider
from .sporttery import SportteryProvider
from .data500 import Data500Provider

__all__ = ["BaseProvider", "CWLProvider", "SportteryProvider", "Data500Provider", "get_provider"]


def get_provider(provider_name: str) -> BaseProvider:
    """根据provider名称获取对应的数据源实例"""
    providers = {
        "cwl": CWLProvider,
        "sporttery": SportteryProvider,
        "data500": Data500Provider,
    }
    cls = providers.get(provider_name)
    if not cls:
        raise ValueError(f"未知的provider: {provider_name}, 可选: {list(providers.keys())}")
    return cls()
