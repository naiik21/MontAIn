"""Rate limiting por IP con ventana deslizante, en memoria.

En memoria a proposito: la demo corre en una sola instancia y asi el
despliegue no arrastra un Redis. Si algun dia se escala a varios workers,
el contador deja de ser global y hay que mover esto a un almacen compartido.
"""

import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """
        Registra un intento de `key`.

        Devuelve (permitido, segundos_para_reintentar). Cuando se permite,
        el segundo valor es 0.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.max_requests:
                retry_after = int(hits[0] + self.window_seconds - now) + 1
                return False, max(retry_after, 1)

            hits.append(now)

            # Evita que el diccionario crezca sin limite con IPs ya caducadas.
            if len(self._hits) > 10_000:
                self._evict_expired(cutoff)

            return True, 0

    def _evict_expired(self, cutoff: float) -> None:
        """Solo se llama con el lock ya tomado."""
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]


def client_ip(request) -> str:
    """
    IP del cliente, respetando el proxy del proveedor de hosting.

    Render, Railway y Vercel terminan TLS por delante de la app, asi que
    request.client.host seria siempre la IP del proxy.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "desconocido"
