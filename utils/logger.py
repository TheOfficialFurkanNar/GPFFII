import json
import time

DEVICE_ID = "node_01"
LOG_FILE  = "log.jsonl"
MIN_LEVEL = "DEBUG"

LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}


class StructuredLogger:
    # all instances (e.g. GPFFIIVisualizer creating its own logger mid-run).
    _start_time = time.time()

    def __init__(self, device_id=DEVICE_ID, log_file=LOG_FILE, min_level=MIN_LEVEL):
        self.device_id = device_id
        self.log_file  = log_file
        self.min_level = min_level
        self._seq = 0

    def _write(self, level, event, data=None, error=None):
        if LEVELS.get(level, 0) < LEVELS.get(self.min_level, 0):
            return

        self._seq += 1

        elapsed = round(time.time() - StructuredLogger._start_time, 6)

        entry = {
            "seq":    self._seq,
            "t":      elapsed,       # relative time shown in CLI AND written to file
            "t_unix": time.time(),   # absolute epoch kept in file for log correlation
            "device": self.device_id,
            "level":  level,
            "event":  event,
        }

        if data  is not None: entry["data"]  = data
        if error is not None: entry["error"] = error

        # ── Transport 1: Console - prints the same entry the file receives,
        # so what you see in the terminal is exactly what's on disk.
        print(json.dumps(entry))

        # ── Transport 2: File - written after print so a crash mid-write
        # never silences the console output.
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # ── Public levelled API ───────────────────────────────────────────────────

    def debug(self, event, data=None):
        self._write("DEBUG", event, data)

    def info(self, event, data=None):
        self._write("INFO", event, data)

    def warning(self, event, data=None):
        self._write("WARNING", event, data)

    def error(self, event, data=None, error=None):
        self._write("ERROR", event, data, error)

    # ── Convenience / domain helpers ─────────────────────────────────────────

    def sensor(self, pin, value, unit=None):
        d = {"pin": pin, "value": value}
        if unit: d["unit"] = unit
        self._write("INFO", "sensor_read", d)

    def state_change(self, from_state, to_state, reason=None):
        d = {"from": from_state, "to": to_state}
        if reason: d["reason"] = reason
        self._write("INFO", "state_change", d)

    def timing(self, label, duration_ms):
        self._write("DEBUG", "timing", {"label": label, "duration_ms": duration_ms})
