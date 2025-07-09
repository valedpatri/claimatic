from functools import wraps
import inspect
from pathlib import Path
import time
from typing import Any
from typing import Callable
from typing import TypeVar

from loguru import logger


T = TypeVar("T")


def log_execution(
    level: str = "INFO",
    track_time: bool = True,
    show_args: bool = False,
    log_exceptions: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to log function execution details using Loguru.

    Args:
        level: Log level as string ("DEBUG", "INFO", "WARNING", etc.)
        track_time: Whether to track and log execution time
        show_args: Whether to include function arguments in the log
        log_exceptions: Whether to log exceptions (with stack trace)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get frame information
            frame = inspect.currentframe()
            if frame is None or frame.f_back is None:
                filename = "unknown"
                lineno = 0
            else:
                frame = frame.f_back
                filename = Path(frame.f_code.co_filename).name
                lineno = frame.f_lineno

            func_name = func.__name__
            arg_str = ""
            if show_args:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
                arg_str = f"({', '.join(args_repr + kwargs_repr)})"

            # Log start
            logger.log(
                level, "START {}:{} - {}{}", filename, lineno, func_name, arg_str
            )
            start = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                if track_time:
                    elapsed = time.perf_counter() - start
                    logger.log(
                        level,
                        "FINISH {}:{} - {}{} in {:.4f}s",
                        filename,
                        lineno,
                        func_name,
                        arg_str,
                        elapsed,
                    )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                if log_exceptions:
                    logger.error(
                        "ERROR {}:{} - {}{} after {:.4f}s: {}",
                        filename,
                        lineno,
                        func_name,
                        arg_str,
                        elapsed,
                        str(e),
                    )
                raise

        return wrapper

    return decorator
