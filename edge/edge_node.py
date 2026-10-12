"""
Three-node test bed for the vehicle trafficability bundle (stdlib sockets only).

    source  holds the jobs: a 400-day weather log + trip requests (the vehicle's sensors)
    infer   fetches a job, forecasts, builds the map, plans the routes (the edge node)
    sink    receives the routes and acknowledges them to the source (the consumer)

Per job, infer reports fetch time (request -> job received) and inference time; the
source measures end-to-end latency (job queued -> ack received) on its own clock, so
the nodes need no clock sync.

Start in this order (each on its own machine, or all on one with 127.0.0.1):

    python edge_node.py sink   --listen 0.0.0.0:6000 --source SOURCE_IP:7000
    python edge_node.py source --listen 0.0.0.0:5001 --ack 0.0.0.0:7000 \\
                               --requests 10,20,30,40,50 --repeats 5 --mode independent
    python edge_node.py infer  --source SOURCE_IP:5001 --sink SINK_IP:6000

    python edge_node.py demo   # all three on this machine, for a quick check

--mode independent: every request carries its own weather log and one trip (worst case,
like one image per request). --mode batched: one job carries the log and all N trips.
Results: source_results.csv (one row per batch) and a summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ framing
def send(sock, obj):
    b = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(b)) + b)
    return len(b) + 4


def recv(sock):
    head = _exact(sock, 4)
    if head is None:
        return None, 0
    n, = struct.unpack(">I", head)
    return json.loads(_exact(sock, n)), n + 4


def _exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _addr(s):
    h, p = s.rsplit(":", 1)
    return h, int(p)


def _listen(addr):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(_addr(addr))
    s.listen(4)
    return s


def _connect(addr, tries=100):
    for _ in range(tries):
        try:
            s = socket.create_connection(_addr(addr))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            time.sleep(0.1)
    raise SystemExit(f"cannot reach {addr}")


# ------------------------------------------------------------------ roles
def run_source(a):
    log_text = (HERE / "sample_weather_log.csv").read_text()
    trips = json.load(open(HERE / "sample_trips.json"))
    pairs = [(t["src"]["cell"], t["dst"]["cell"]) for t in trips]
    queue, lock = deque(), threading.Lock()
    acks, done, ready = {}, threading.Event(), threading.Event()

    def ack_server():
        srv = _listen(a.ack)
        conn, _ = srv.accept()
        while True:
            msg, _ = recv(conn)
            if msg is None:
                return
            with lock:
                acks[msg["job"]] = (time.perf_counter(), msg)

    def fetch_server():
        srv = _listen(a.listen)
        conn, _ = srv.accept()                    # infer connects once its models are loaded
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ready.set()
        while True:
            msg, _ = recv(conn)
            if msg is None:
                return
            with lock:
                job = queue.popleft() if queue else None
            if job is not None:
                send(conn, job)
            else:
                send(conn, {"type": "done" if done.is_set() else "wait"})

    threading.Thread(target=ack_server, daemon=True).start()
    threading.Thread(target=fetch_server, daemon=True).start()
    print(f"source: serving jobs on {a.listen}, acks on {a.ack}; waiting for infer", flush=True)
    ready.wait()

    rows, job_id, k = [], 0, 0
    for n in [int(x) for x in a.requests.split(",")]:
        for rep in range(a.repeats):
            if a.mode == "independent":
                jobs = [{"type": "job", "job": job_id + i, "log": log_text, "band": a.band, "h": a.h,
                         "trips": [pairs[(k + i) % len(pairs)]]} for i in range(n)]
            else:
                jobs = [{"type": "job", "job": job_id, "log": log_text, "band": a.band, "h": a.h,
                         "trips": [pairs[(k + i) % len(pairs)] for i in range(n)]}]
            ids = [j["job"] for j in jobs]
            job_id += len(jobs)
            k += n
            t0 = time.perf_counter()
            with lock:
                queue.extend(jobs)
            while True:
                with lock:
                    if all(i in acks for i in ids):
                        break
                time.sleep(0.0005)
            with lock:
                got = [acks.pop(i) for i in ids]
            e2e = max(t for t, _ in got) - t0
            rows.append(dict(requests=n, mode=a.mode, repeat=rep,
                             fetch_s=sum(m["fetch_s"] for _, m in got),
                             inference_s=sum(m["infer_s"] for _, m in got),
                             e2e_ms=e2e * 1000,
                             bytes_up=sum(m["bytes_in"] for _, m in got),
                             bytes_down=sum(m["bytes_out"] for _, m in got)))
            print(f"  n={n:<3} rep={rep}  fetch {rows[-1]['fetch_s']:.3f}s  "
                  f"inference {rows[-1]['inference_s']:.3f}s  e2e {rows[-1]['e2e_ms']:.1f}ms", flush=True)
    done.set()

    out = HERE / f"source_results_{a.mode}.csv"
    with open(out, "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(f"\n{'Requests':>8} {'FT (s)':>9} {'Inference (s)':>14} {'E2E (ms)':>10}   mode={a.mode}")
    for n in sorted({r["requests"] for r in rows}):
        g = [r for r in rows if r["requests"] == n]
        mean = lambda key: sum(r[key] for r in g) / len(g)
        print(f"{n:>8} {mean('fetch_s'):>9.3f} {mean('inference_s'):>14.3f} {mean('e2e_ms'):>10.1f}")
    print(f"-> {out.name}")
    time.sleep(0.5)


def run_infer(a):
    sys.path.insert(0, str(HERE))
    import io
    import trafficability_edge as E
    fc_model = E.Forecaster(HERE / "vehicle_trafficability_forecaster.pkl", backend=a.backend)
    pl = E.Planner(HERE / "vehicle_trafficability_planner.pkl")
    src, sink = _connect(a.source), _connect(a.sink)
    print(f"infer: ready ({fc_model.backend}), fetching from {a.source}, sending to {a.sink}", flush=True)
    while True:
        t0 = time.perf_counter()
        send(src, {"type": "fetch"})
        job, n_in = recv(src)
        if job is None or job["type"] == "done":
            break
        if job["type"] == "wait":
            time.sleep(0.002)
            continue
        t1 = time.perf_counter()
        log = E.read_log(io.StringIO(job["log"]))
        fc, rci, res = E.end_to_end(fc_model, pl, log, job["band"], [tuple(t) for t in job["trips"]], job["h"])
        result = {"job": job["job"], "forecast": E.Forecaster.message(fc).hex(),
                  "status": fc_model.assess(fc, band=job["band"]),
                  "routes": [None if p is None else p.tolist() for p, _ in res]}
        t2 = time.perf_counter()
        n_out = send(sink, {"job": job["job"], "result": result, "fetch_s": t1 - t0,
                            "infer_s": t2 - t1, "bytes_in": n_in})
    print("infer: done", flush=True)


def run_sink(a):
    srv = _listen(a.listen)
    print(f"sink: listening on {a.listen}", flush=True)
    conn, _ = srv.accept()
    back = None
    while True:
        msg, n = recv(conn)
        if msg is None:
            break
        back = back or _connect(a.source)
        send(back, {"job": msg["job"], "fetch_s": msg["fetch_s"], "infer_s": msg["infer_s"],
                    "bytes_in": msg["bytes_in"], "bytes_out": n})
    print("sink: done", flush=True)


def run_demo(a):
    py = sys.executable
    me = str(Path(__file__).resolve())
    extra = ["--backend", a.backend]
    sink = subprocess.Popen([py, me, "sink", "--listen", "127.0.0.1:6000", "--source", "127.0.0.1:7000"])
    time.sleep(0.5)
    source = subprocess.Popen([py, me, "source", "--listen", "127.0.0.1:5001", "--ack", "127.0.0.1:7000",
                               "--requests", a.requests, "--repeats", str(a.repeats), "--mode", a.mode])
    time.sleep(0.5)
    infer = subprocess.Popen([py, me, "infer", "--source", "127.0.0.1:5001", "--sink", "127.0.0.1:6000"] + extra)
    source.wait()
    for p in (infer, sink):
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=["source", "infer", "sink", "demo"])
    ap.add_argument("--listen", default="0.0.0.0:5001")
    ap.add_argument("--ack", default="0.0.0.0:7000")
    ap.add_argument("--source", default="127.0.0.1:5001")
    ap.add_argument("--sink", default="127.0.0.1:6000")
    ap.add_argument("--requests", default="10,20,30,40,50")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mode", choices=["independent", "batched"], default="independent")
    ap.add_argument("--band", default="D")
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, str(a.threads))
    {"source": run_source, "infer": run_infer, "sink": run_sink, "demo": run_demo}[a.role](a)
