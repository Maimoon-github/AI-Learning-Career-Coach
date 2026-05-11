"""OpenTelemetry setup."""

# src/observability/tracer.py

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import Counter, Histogram, start_http_server


# ── Prometheus metrics ─────────────────────────────────────────
AGENT_CALLS = Counter("agent_calls_total", "Total agent invocations", ["agent_name"])
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent latency", ["agent_name"])
LLM_TOKENS = Counter("llm_tokens_total", "Total LLM tokens used", ["model", "direction"])
EVAL_SCORES = Histogram("eval_scores", "Evaluator scores", buckets=[0.1*i for i in range(11)])


def setup_observability():
    """Initialize tracing and metrics. Call once at application start."""
    # OpenTelemetry
    provider = TracerProvider()
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        exporter = OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Prometheus metrics server
    metrics_port = int(os.environ.get("PROMETHEUS_PORT", "9090"))
    start_http_server(metrics_port)
    print(f"📊 Metrics server started on :{metrics_port}")


def get_tracer():
    return trace.get_tracer("ai-coach")