import asyncio
import json
import os
import socket
import time
from collections import deque
from datetime import datetime

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# GPU detection (graceful fallback when no NVIDIA GPU present)
# ---------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
    GPU_NAMES: list[str] = []
    for _i in range(GPU_COUNT):
        _h = pynvml.nvmlDeviceGetHandleByIndex(_i)
        _n = pynvml.nvmlDeviceGetName(_h)
        GPU_NAMES.append(_n.decode() if isinstance(_n, bytes) else _n)
except Exception:
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    GPU_NAMES = []

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL = 2        # seconds between samples
MAX_SAMPLES = 1800         # 1 hour @ 2 s intervals
HOST_ROOT = os.environ.get("HOST_ROOT", "/")
HOSTNAME = socket.gethostname()

# ---------------------------------------------------------------------------
# Ring-buffer history
# ---------------------------------------------------------------------------
history: dict[str, deque] = {
    "timestamps": deque(maxlen=MAX_SAMPLES),
    "cpu":        deque(maxlen=MAX_SAMPLES),
    "memory":     deque(maxlen=MAX_SAMPLES),
    "net_sent":   deque(maxlen=MAX_SAMPLES),
    "net_recv":   deque(maxlen=MAX_SAMPLES),
    "disk_read":  deque(maxlen=MAX_SAMPLES),
    "disk_write": deque(maxlen=MAX_SAMPLES),
    "disk_usage": deque(maxlen=MAX_SAMPLES),
}
if GPU_AVAILABLE:
    history["gpu_util"] = [deque(maxlen=MAX_SAMPLES) for _ in range(GPU_COUNT)]
    history["gpu_mem"]  = [deque(maxlen=MAX_SAMPLES) for _ in range(GPU_COUNT)]

# Previous counters for rate calculation
_prev_net  = psutil.net_io_counters()
_prev_disk = psutil.disk_io_counters()
_prev_time = time.monotonic()

# Active WebSocket connections
clients: set[WebSocket] = set()

# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_metrics() -> dict:
    global _prev_net, _prev_disk, _prev_time

    now     = time.monotonic()
    elapsed = max(now - _prev_time, 0.001)
    _prev_time = now

    # CPU (non-blocking; first call after startup may return 0.0 — acceptable)
    cpu = psutil.cpu_percent()

    # Memory
    mem = psutil.virtual_memory()

    # Network rates (bytes/s)
    net = psutil.net_io_counters()
    net_sent = max(0.0, (net.bytes_sent - _prev_net.bytes_sent) / elapsed)
    net_recv = max(0.0, (net.bytes_recv - _prev_net.bytes_recv) / elapsed)
    _prev_net = net

    # Disk I/O rates (bytes/s)
    try:
        disk_io   = psutil.disk_io_counters()
        disk_read  = max(0.0, (disk_io.read_bytes  - _prev_disk.read_bytes)  / elapsed)
        disk_write = max(0.0, (disk_io.write_bytes - _prev_disk.write_bytes) / elapsed)
        _prev_disk = disk_io
    except Exception:
        disk_read = disk_write = 0.0

    # Disk usage %
    try:
        disk_usage = psutil.disk_usage(HOST_ROOT).percent
    except Exception:
        disk_usage = psutil.disk_usage("/").percent

    # GPU
    gpus: list[dict] = []
    if GPU_AVAILABLE:
        for i in range(GPU_COUNT):
            try:
                handle   = pynvml.nvmlDeviceGetHandleByIndex(i)
                util     = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp     = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpus.append({
                    "name":      GPU_NAMES[i],
                    "util":      util.gpu,
                    "mem_pct":   round(mem_info.used / mem_info.total * 100, 1),
                    "mem_used":  mem_info.used,
                    "mem_total": mem_info.total,
                    "temp":      temp,
                })
            except Exception:
                gpus.append({"name": GPU_NAMES[i], "util": 0, "mem_pct": 0.0,
                             "mem_used": 0, "mem_total": 0, "temp": 0})

    ts = datetime.now().isoformat()

    # Append to ring buffers
    history["timestamps"].append(ts)
    history["cpu"].append(round(cpu, 1))
    history["memory"].append(round(mem.percent, 1))
    history["net_sent"].append(round(net_sent, 1))
    history["net_recv"].append(round(net_recv, 1))
    history["disk_read"].append(round(disk_read, 1))
    history["disk_write"].append(round(disk_write, 1))
    history["disk_usage"].append(round(disk_usage, 1))
    if GPU_AVAILABLE:
        for i, g in enumerate(gpus):
            history["gpu_util"][i].append(g["util"])
            history["gpu_mem"][i].append(g["mem_pct"])

    return {
        "timestamp":  ts,
        "cpu":        cpu,
        "cpu_cores":  psutil.cpu_count(logical=True),
        "memory":     {"percent": mem.percent, "used": mem.used, "total": mem.total},
        "net_sent":   net_sent,
        "net_recv":   net_recv,
        "disk_read":  disk_read,
        "disk_write": disk_write,
        "disk_usage": disk_usage,
        "disk_total": psutil.disk_usage(HOST_ROOT).total,
        "gpus":       gpus,
        "hostname":   HOSTNAME,
    }


# ---------------------------------------------------------------------------
# Background collector + broadcaster
# ---------------------------------------------------------------------------

async def background_collector():
    while True:
        metrics = collect_metrics()
        if clients:
            msg  = json.dumps({"type": "metrics", "data": metrics})
            dead: set[WebSocket] = set()
            for ws in list(clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            for _dead_ws in dead:
                clients.discard(_dead_ws)
        await asyncio.sleep(SAMPLE_INTERVAL)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Resource Dashboard")


@app.on_event("startup")
async def startup():
    # Warm up cpu_percent so the first real reading isn't 0.0
    psutil.cpu_percent()
    asyncio.create_task(background_collector())


@app.get("/api/history")
async def get_history():
    result: dict = {
        "timestamps":   list(history["timestamps"]),
        "cpu":          list(history["cpu"]),
        "memory":       list(history["memory"]),
        "net_sent":     list(history["net_sent"]),
        "net_recv":     list(history["net_recv"]),
        "disk_read":    list(history["disk_read"]),
        "disk_write":   list(history["disk_write"]),
        "disk_usage":   list(history["disk_usage"]),
        "gpu_available": GPU_AVAILABLE,
        "gpu_names":    GPU_NAMES,
        "hostname":     HOSTNAME,
    }
    if GPU_AVAILABLE:
        result["gpu_util"] = [list(q) for q in history["gpu_util"]]
        result["gpu_mem"]  = [list(q) for q in history["gpu_mem"]]
    return result


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            # Keep the connection alive; client pings aren't required
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())
