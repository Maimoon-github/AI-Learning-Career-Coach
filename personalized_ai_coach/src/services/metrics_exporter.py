from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Node execution metrics
node_latency = Histogram(
    "langgraph_node_duration_seconds",
    "Node execution time",
    ["node_name", "status"]
)
node_errors = Counter(
    "langgraph_node_errors_total",
    "Node error count",
    ["node_name", "error_type"]
)

# Crew metrics
crew_success = Counter(
    "crew_success_total",
    "Successful crew executions",
    ["crew_name"]
)
crew_failure = Counter(
    "crew_failure_total",
    "Failed crew executions",
    ["crew_name", "error_type"]
)

# LLM metrics
llm_token_usage = Gauge(
    "llm_token_usage",
    "Tokens consumed per request",
    ["model", "task_type"]
)
llm_latency = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration",
    ["model", "task_type"]
)

# Tool metrics
tool_call_count = Counter(
    "tool_calls_total",
    "Tool invocation count",
    ["tool_name", "status"]
)
tool_latency = Histogram(
    "tool_duration_seconds",
    "Tool execution time",
    ["tool_name"]
)

# HITL metrics
hitl_pause_duration = Histogram(
    "hitl_pause_seconds",
    "Time spent in HITL interrupt",
    ["decision"]
)

# Checkpoint metrics
checkpoint_write_latency = Histogram(
    "checkpoint_write_seconds",
    "Time to write checkpoint"
)
checkpoint_read_latency = Histogram(
    "checkpoint_read_seconds",
    "Time to read checkpoint"
)

def start_metrics_server(port: int = 9090):
    start_http_server(port)