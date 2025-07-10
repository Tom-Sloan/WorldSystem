"""WorldSystem Common Package - Shared utilities and configurations."""

__version__ = "1.0.0"

from .rabbitmq_config import (
    EXCHANGES,
    ROUTING_KEYS,
    declare_exchanges,
    get_exchange_config
)

__all__ = [
    'EXCHANGES',
    'ROUTING_KEYS', 
    'declare_exchanges',
    'get_exchange_config'
]