"""
Voice Phishing Detection - Langfuse Monitoring Integration
"""
import time
import functools
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LangfuseMonitor:
    """Langfuse integration for LLMOps monitoring"""
    
    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True
    ):
        self.enabled = enabled and public_key and secret_key
        self.langfuse = None
        
        if self.enabled:
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                logger.info("Langfuse monitoring initialized")
            except ImportError:
                logger.warning("Langfuse not installed. Monitoring disabled.")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.enabled = False
    
    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """Create a new trace"""
        if not self.enabled:
            return DummyTrace()
        
        return self.langfuse.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or []
        )
    
    def log_generation(
        self,
        trace,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log LLM generation to trace"""
        if not self.enabled:
            return
        
        trace.generation(
            name=name,
            model=model,
            input=prompt,
            output=completion,
            usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            metadata={
                "latency_ms": latency_ms,
                **(metadata or {})
            }
        )
    
    def log_span(
        self,
        trace,
        name: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a span (non-LLM operation) to trace"""
        if not self.enabled:
            return
        
        trace.span(
            name=name,
            input=input_data,
            output=output_data,
            metadata={
                "latency_ms": latency_ms,
                **(metadata or {})
            }
        )
    
    def log_score(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """Log a score/feedback to trace"""
        if not self.enabled:
            return
        
        trace.score(
            name=name,
            value=value,
            comment=comment
        )
    
    def log_event(
        self,
        trace,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an event to trace"""
        if not self.enabled:
            return
        
        trace.event(
            name=name,
            metadata=metadata or {}
        )
    
    def flush(self):
        """Flush pending logs"""
        if self.enabled and self.langfuse:
            self.langfuse.flush()


class DummyTrace:
    """Dummy trace for when Langfuse is disabled"""
    
    def generation(self, **kwargs):
        pass
    
    def span(self, **kwargs):
        pass
    
    def score(self, **kwargs):
        pass
    
    def event(self, **kwargs):
        pass


def track_latency(func: Callable) -> Callable:
    """Decorator to track function latency"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000
        
        # Attach latency to result if possible
        if hasattr(result, '__dict__'):
            result._latency_ms = latency_ms
        
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000
        
        if hasattr(result, '__dict__'):
            result._latency_ms = latency_ms
        
        return result
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "detection_latency_ms": [],
            "rag_latency_ms": [],
            "total_latency_ms": [],
            "tokens_used": [],
            "risk_level_counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        }
        self.start_time = datetime.now()
    
    def record_request(self, success: bool = True):
        """Record a request"""
        self.metrics["requests_total"] += 1
        if success:
            self.metrics["requests_success"] += 1
        else:
            self.metrics["requests_error"] += 1
    
    def record_latency(self, category: str, latency_ms: float):
        """Record latency for a category"""
        key = f"{category}_latency_ms"
        if key in self.metrics:
            self.metrics[key].append(latency_ms)
            # Keep only last 1000 samples
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
    
    def record_tokens(self, tokens: int):
        """Record tokens used"""
        self.metrics["tokens_used"].append(tokens)
        if len(self.metrics["tokens_used"]) > 1000:
            self.metrics["tokens_used"] = self.metrics["tokens_used"][-1000:]
    
    def record_risk_level(self, level: str):
        """Record risk level"""
        if level in self.metrics["risk_level_counts"]:
            self.metrics["risk_level_counts"][level] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        import numpy as np
        
        def calc_stats(values):
            if not values:
                return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99))
            }
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "requests": {
                "total": self.metrics["requests_total"],
                "success": self.metrics["requests_success"],
                "error": self.metrics["requests_error"],
                "success_rate": (
                    self.metrics["requests_success"] / max(self.metrics["requests_total"], 1)
                )
            },
            "latency": {
                "detection": calc_stats(self.metrics["detection_latency_ms"]),
                "rag": calc_stats(self.metrics["rag_latency_ms"]),
                "total": calc_stats(self.metrics["total_latency_ms"])
            },
            "tokens": calc_stats(self.metrics["tokens_used"]),
            "risk_distribution": self.metrics["risk_level_counts"]
        }


# Global instances
_monitor: Optional[LangfuseMonitor] = None
_metrics: Optional[MetricsCollector] = None


def get_monitor() -> LangfuseMonitor:
    """Get or create global monitor"""
    global _monitor
    if _monitor is None:
        from src.config import settings
        _monitor = LangfuseMonitor(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            enabled=settings.LANGFUSE_ENABLED
        )
    return _monitor


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
