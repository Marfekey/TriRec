
import ctypes
import functools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# --- Parallelism ---
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 36))
API_BATCH = int(os.getenv("API_BATCH", 32))

_API_KEY = os.getenv("OPENAI_API_KEY", "")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None

client = (
    OpenAI(api_key=_API_KEY, base_url=_BASE_URL) if _BASE_URL else OpenAI(api_key=_API_KEY)
)


# --- Terminable thread (used by the timeout decorator) ---
class _ThreadKiller(threading.Thread):
    def __init__(self, target, exc_cls, repeat_sec: float = 2.0):
        super().__init__(daemon=True)
        self._target_thread = target
        self._exc_cls = exc_cls
        self._repeat_sec = repeat_sec

    def run(self):
        while self._target_thread.is_alive():
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(self._target_thread.ident),
                ctypes.py_object(self._exc_cls),
            )
            self._target_thread.join(self._repeat_sec)


class _TerminableThread(threading.Thread):
    def terminate(self, exc_cls, repeat_sec: float = 2.0):
        if not self.is_alive():
            return
        _ThreadKiller(self, exc_cls, repeat_sec).start()


def timeout(sec: int, repeat_sec: float = 1.0):
    """Function-level timeout decorator: raise TimeoutError after `sec` seconds."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            class _FuncTimeout(TimeoutError):
                pass

            result, exception = [], []

            def _run():
                try:
                    result.append(func(*args, **kwargs))
                except _FuncTimeout:
                    pass
                except Exception as e:
                    exception.append(e)

            t = _TerminableThread(target=_run, daemon=True)
            t.start()
            t.join(timeout=sec)
            if t.is_alive():
                t.terminate(_FuncTimeout, repeat_sec)
                raise TimeoutError(f"Function {func.__name__} timed out after {sec}s")
            if exception:
                raise exception[0]
            return result[0] if result else None

        return wrapper
    return decorator


# --- LLM chat ---
@timeout(100)
def get_response_from_openai(prompt: str, model: str, api_key: str | None = None) -> str | None:
    """Single-turn chat; retry up to 10 times on failure. The `api_key` argument is kept for backward compatibility and is ignored."""
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1, 11):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=800, n=1, temperature=0.7,
            )
            if resp.choices and resp.choices[0].message.content:
                content = resp.choices[0].message.content.strip()
                if content:
                    return content
            print(f"[warn] Model returned empty content; retry {attempt}/10")
        except Exception as e:
            print(f"[warn] LLM request failed: {e}; retry {attempt}/10")
        time.sleep(2)
    print("[error] LLM request reached the maximum retry count")
    return None


def parallel_get_responses(prompts: list, model: str, max_workers: int = MAX_WORKERS) -> list:
    """Chat in parallel."""
    if not prompts:
        return []
    max_workers = min(max_workers, MAX_WORKERS)
    results: list = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(get_response_from_openai, p, model) for p in prompts]
        for i, fut in enumerate(futures):
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"[warn] Parallel LLM request {i} failed: {e}")
    return results


# --- Embedding ---
@timeout(200)
def get_embedding_from_openai(
    text: str, model: str = "text-embedding-ada-002", api_key: str | None = None
):
    for attempt in range(1, 11):
        try:
            resp = client.embeddings.create(input=text, model=model)
            return resp.data[0].embedding
        except Exception as e:
            print(f"[warn] Embedding request failed: {e}; retry {attempt}/10")
            time.sleep(2)
    print("[error] Embedding request reached the maximum retry count")
    return None


def parallel_get_embeddings(
    texts: list, model: str = "text-embedding-ada-002", max_workers: int = MAX_WORKERS
) -> list:
    if not texts:
        return []
    max_workers = min(max_workers, MAX_WORKERS)
    results: list = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(get_embedding_from_openai, t, model) for t in texts]
        for i, fut in enumerate(futures):
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"[warn] Parallel Embedding request {i} failed: {e}")
    return results
