"""Cache LRU en memoria para los resultados de /process-gpx.

La clave es el hash SHA-256 del archivo subido: el mismo GPX byte a byte
produce el mismo analisis, asi que repetirlo solo quema CPU y una llamada a
Claude. Esto importa en una demo publica, donde lo normal es que mucha gente
pruebe con los mismos archivos de ejemplo.

En memoria a proposito, igual que el rate limiter: una sola instancia, cero
infraestructura. Si se escala a varios workers, cada uno tendra su propia
cache (funciona, solo pierde eficiencia) y seria el momento de un Redis.
"""

import threading
from collections import OrderedDict


class LRUCache:
    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: dict) -> None:
        if self.max_entries <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
