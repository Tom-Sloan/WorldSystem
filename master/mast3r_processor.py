#!/usr/bin/env python3
"""
mast3r_processor.py
───────────────────
Consumes RGB frames from RabbitMQ (exactly like slam3r_processor.py) and streams
MASt3R-SLAM results live to Rerun.  It **does not** publish pose / point-cloud
messages back to RabbitMQ – all visualisation is handled via Rerun.

Usage inside the container is automatic (see Dockerfile / docker-compose.yml).
"""

import asyncio, logging, os, time
from datetime import datetime
from pathlib import Path

import aio_pika, cv2, numpy as np, torch, yaml, rerun as rr

# ─────────────────────────── env / defaults ────────────────────────────
RABBITMQ_URL             = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
VIDEO_FRAMES_EXCHANGE    = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
RESTART_EXCHANGE         = os.getenv("RESTART_EXCHANGE", "restart_exchange")

RERUN_ENABLED            = os.getenv("RERUN_ENABLED", "true").lower() == "true"
RERUN_CONNECT_URL        = os.getenv("RERUN_CONNECT_URL", "rerun+http://127.0.0.1:9876/proxy")

CFG_FILE_PATH            = os.getenv("MAST3R_CONFIG_FILE", "/app/config/base.yaml")
CHECKPOINTS_DIR          = os.getenv("MAST3R_CHECKPOINTS_DIR", "/app/checkpoints")

TARGET_IMG_SIZE          = int(os.getenv("TARGET_IMAGE_SIZE", "512"))   # MASt3R default

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── logging setup ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("mast3r_processor")

# ─────────────────── import MASt3R-SLAM internals ──────────────────────
log.info("Importing MASt3R-SLAM…asdfasdf")
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.frame  import Mode, create_frame, SharedKeyframes, SharedStates
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono

# Disable multi-processing inside MASt3R for simplicity
config["single_thread"] = True

# ─────────────────────────── Rerun init ────────────────────────────────
rerun_connected = False
if RERUN_ENABLED:
    rr.init("MASt3R_Processor", spawn=False)
    try:
        rr.connect_grpc(RERUN_CONNECT_URL, flush_timeout_sec=15)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        rerun_connected = True
        log.info("Connected to Rerun viewer.")
    except Exception as e:
        log.warning("Rerun connection failed: %s", e)

# ─────────────── Global SLAM state objects (single-thread) ─────────────
manager   = torch.multiprocessing.Manager()
keyframes = states = tracker = factor_graph = None
camera_positions: list[np.ndarray] = []

def initialise_slam():
    """Lazy-initialise MASt3R-SLAM objects once the first frame arrives."""
    global keyframes, states, tracker, factor_graph
    if keyframes is not None:
        return

    log.info("Loading configuration %s", CFG_FILE_PATH)
    load_config(CFG_FILE_PATH)
    h = w = TARGET_IMG_SIZE
    keyframes = SharedKeyframes(manager, h, w, device=device)
    states    = SharedStates(manager, h, w, device=device)
    model     = load_mast3r(device=device)
    tracker   = FrameTracker(model, keyframes, device)
    K         = None  # we run without explicit calibration here
    factor_graph = FactorGraph(model, keyframes, K, device)
    log.info("MASt3R-SLAM initialised on %s.", device)

# ───────────────────── image pre-processing helper ─────────────────────
def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    return torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

def cv_to_rerun_xyz(xyz: np.ndarray) -> np.ndarray:
    """convert camera-coords (x-right,y-down,z-forward) → Rerun RHS-Y-Up world."""
    xyz = xyz.copy()
    xyz[:, 0] *= -1
    xyz[:, 1] *= -1
    return xyz

def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    q = np.empty(4); t = np.trace(m)
    if t > 0:
        t = np.sqrt(t + 1); q[3] = 0.5 * t; t = 0.5 / t
        q[0] = (m[2,1]-m[1,2])*t; q[1] = (m[0,2]-m[2,0])*t; q[2] = (m[1,0]-m[0,1])*t
    else:
        i = np.argmax(np.diag(m)); j, k = (i+1)%3, (i+2)%3
        t = np.sqrt(m[i,i]-m[j,j]-m[k,k]+1)
        q[i] = 0.5*t; t = 0.5/t
        q[3] = (m[k,j]-m[j,k])*t; q[j] = (m[j,i]+m[i,j])*t; q[k] = (m[k,i]+m[i,k])*t
    return q

# ───────────────────────── frame-level SLAM ────────────────────────────
frame_idx = 0
def process_frame(img_bgr: np.ndarray, ts_ns: int):
    global frame_idx
    if keyframes is None:
        initialise_slam()

    tensor = preprocess(img_bgr).to(device)
    frame  = create_frame(frame_idx, {"img": tensor, "unnormalized_img": img_bgr,
                                      "true_shape": [TARGET_IMG_SIZE, TARGET_IMG_SIZE]},
                          torch.eye(4, device=device), TARGET_IMG_SIZE, device)

    if states.get_mode() == Mode.INIT:
        # Bootstrap with mono inference
        X, C = mast3r_inference_mono(tracker.model, frame)
        frame.update_pointmap(X, C)
        keyframes.append(frame)
        states.set_frame(frame)
        states.set_mode(Mode.TRACKING)
        log.info("Bootstrap complete.")
    else:
        # Normal tracking step
        add_kf, *_ = tracker.track(frame)
        states.set_frame(frame)
        if add_kf:
            keyframes.append(frame)

    # ───────── Rerun visualisation ─────────
    if rerun_connected and frame.X_canon is not None:
        xyz = frame.X_canon.cpu().numpy().reshape(-1, 3)
        C   = frame.C.cpu().numpy().reshape(-1)
        mask = C > 0.5
        if mask.any():
            rr.log("world/points",
                   rr.Points3D(positions=cv_to_rerun_xyz(xyz[mask].astype(np.float32)),
                               colors=np.repeat(np.array([[0, 255, 0]], np.uint8),
                                                mask.sum(), axis=0),
                               radii=np.full(mask.sum(), 0.006, np.float32)))
    # Pose logging
    if rerun_connected:
        T = frame.T_WC.matrix()[0].cpu().numpy()
        rr.log("world/camera",
               rr.Transform3D(translation=T[:3, 3],
                              rotation=rr.Quaternion(xyzw=matrix_to_quat(T[:3, :3]))))
        camera_positions.append(T[:3, 3].copy())
        if len(camera_positions) > 1:
            rr.log("world/camera_path",
                   rr.LineStrips3D(np.stack(camera_positions, dtype=np.float32)[None]))
    frame_idx += 1

# ───────────────────── RabbitMQ callbacks / consumer ───────────────────
async def on_frame(msg: aio_pika.IncomingMessage):
    async with msg.process():
        try:
            ts_ns = int(msg.headers.get("timestamp_ns", "0"))
            img   = cv2.imdecode(np.frombuffer(msg.body, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("Image decode failed.")
                return
            t0 = time.time()
            process_frame(img, ts_ns)
            log.info("Frame %d processed in %.3fs", frame_idx - 1, time.time() - t0)
        except Exception:
            log.exception("Error in frame handler")

async def on_restart(msg: aio_pika.IncomingMessage):
    async with msg.process():
        global keyframes, states, tracker, factor_graph, camera_positions, frame_idx
        log.info("Restart signal received – resetting state.")
        keyframes = states = tracker = factor_graph = None
        camera_positions.clear()
        frame_idx = 0

async def main():
    connection = await aio_pika.connect_robust(RABBITMQ_URL, heartbeat=60)
    async with connection:
        ch = await connection.channel()
        await ch.set_qos(prefetch_count=1)

        ex_frames  = await ch.declare_exchange(VIDEO_FRAMES_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
        ex_restart = await ch.declare_exchange(RESTART_EXCHANGE,      aio_pika.ExchangeType.FANOUT, durable=True)

        q_frames   = await ch.declare_queue("mast3r_video_frames_queue", durable=True)
        q_restart  = await ch.declare_queue("mast3r_restart_queue",      durable=True)
        await q_frames.bind(ex_frames)
        await q_restart.bind(ex_restart)

        await q_frames.consume(on_frame)
        await q_restart.consume(on_restart)

        log.info("MASt3R processor ready – awaiting frames…")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("User interrupted – shutting down")