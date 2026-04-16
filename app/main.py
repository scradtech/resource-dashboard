import asyncio
import json
import os
import socket
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import docker
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
# Docker container stats
# ---------------------------------------------------------------------------
_docker_client = None
_prev_ctr_counters: dict = {}   # container_id -> {net_rx, net_tx, blk_r, blk_w, ts}
_latest_containers: list[dict] = []
_container_executor = ThreadPoolExecutor(max_workers=20)


def _get_docker_client():
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
    return _docker_client


def _fetch_container_stats(c) -> dict | None:
    """Fetch stats for a single container (runs in a thread)."""
    try:
        s = c.stats(stream=False)

        # CPU %
        cpu_delta = (s["cpu_stats"]["cpu_usage"]["total_usage"]
                     - s["precpu_stats"]["cpu_usage"]["total_usage"])
        sys_delta  = (s["cpu_stats"].get("system_cpu_usage", 0)
                     - s["precpu_stats"].get("system_cpu_usage", 0))
        num_cpus   = s["cpu_stats"].get("online_cpus") or len(
                         s["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
        cpu_pct = (cpu_delta / sys_delta * num_cpus * 100.0) if sys_delta > 0 else 0.0

        # Memory
        mem_stats  = s.get("memory_stats", {})
        # Exclude page cache so the number matches `docker stats`
        cache      = (mem_stats.get("stats") or {}).get("inactive_file", 0)
        mem_usage  = max(0, mem_stats.get("usage", 0) - cache)
        mem_limit  = mem_stats.get("limit", 1) or 1
        mem_pct    = mem_usage / mem_limit * 100

        # Cumulative network counters
        net_rx = net_tx = 0
        for iface in (s.get("networks") or {}).values():
            net_rx += iface.get("rx_bytes", 0)
            net_tx += iface.get("tx_bytes", 0)

        # Cumulative block I/O counters
        blk_r = blk_w = 0
        for entry in (s.get("blkio_stats", {}).get("io_service_bytes_recursive") or []):
            op = entry.get("op", "").lower()
            if op == "read":
                blk_r += entry.get("value", 0)
            elif op == "write":
                blk_w += entry.get("value", 0)

        # Rate calculation using previous sample
        now  = time.monotonic()
        prev = _prev_ctr_counters.get(c.id, {})
        dt   = max(now - prev.get("ts", now - SAMPLE_INTERVAL), 0.001)
        net_rx_rate = max(0.0, (net_rx - prev.get("net_rx", net_rx)) / dt)
        net_tx_rate = max(0.0, (net_tx - prev.get("net_tx", net_tx)) / dt)
        blk_r_rate  = max(0.0, (blk_r  - prev.get("blk_r",  blk_r))  / dt)
        blk_w_rate  = max(0.0, (blk_w  - prev.get("blk_w",  blk_w))  / dt)
        _prev_ctr_counters[c.id] = {"ts": now, "net_rx": net_rx, "net_tx": net_tx,
                                     "blk_r": blk_r, "blk_w": blk_w}

        tags  = c.image.tags
        image = tags[0] if tags else (c.image.short_id or "")

        return {
            "id":        c.short_id,
            "name":      c.name,
            "image":     image,
            "status":    c.status,
            "cpu_pct":   round(cpu_pct, 1),
            "mem_usage": mem_usage,
            "mem_limit": mem_limit,
            "mem_pct":   round(mem_pct, 1),
            "net_rx":    round(net_rx_rate, 1),
            "net_tx":    round(net_tx_rate, 1),
            "blk_r":     round(blk_r_rate, 1),
            "blk_w":     round(blk_w_rate, 1),
        }
    except Exception:
        return None


def _collect_containers_sync() -> list[dict]:
    try:
        containers = _get_docker_client().containers.list()
        futures    = [_container_executor.submit(_fetch_container_stats, c) for c in containers]
        results    = [f.result() for f in futures]
        return sorted([r for r in results if r], key=lambda x: x["name"])
    except Exception:
        return []


async def background_container_collector():
    loop = asyncio.get_event_loop()
    while True:
        global _latest_containers
        try:
            _latest_containers = await loop.run_in_executor(None, _collect_containers_sync)
        except Exception:
            pass
        await asyncio.sleep(SAMPLE_INTERVAL)

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
        "containers": _latest_containers,
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
    asyncio.create_task(background_container_collector())


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
        "containers":   _latest_containers,
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
