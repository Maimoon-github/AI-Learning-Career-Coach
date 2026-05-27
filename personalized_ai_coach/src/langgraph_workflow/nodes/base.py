from functools import wraps
import time
from src.services.metrics_exporter import node_latency, node_errors
from src.utils.structlog_config import bind_context

def track_node(node_name):
    def decorator(func):
        @wraps(func)
        async def wrapper(state):
            bind_context(node_name=node_name)
            start = time.perf_counter()
            try:
                result = await func(state)
                duration = time.perf_counter() - start
                node_latency.labels(node_name=node_name, status="success").observe(duration)
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                node_latency.labels(node_name=node_name, status="error").observe(duration)
                node_errors.labels(node_name=node_name, error_type=type(e).__name__).inc()
                raise
        return wrapper
    return decorator