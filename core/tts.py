"""core/tts.py - 通过 subprocess 调用 tts conda 环境的 worker"""
from __future__ import annotations
import subprocess, json, threading, os, time, logging, tempfile

logger    = logging.getLogger(__name__)
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_PY = os.path.join(BASE_DIR, 'tts_worker.py')
TTS_PY = '/home/zsy/anaconda3/envs/tts/bin/python3'

_proc : subprocess.Popen | None = None
_lock = threading.Lock()
_ready = False

def _start_worker():
    global _proc, _ready
    logger.info("启动 TTS worker...")
    _proc = subprocess.Popen(
        [TTS_PY, WORKER_PY],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
        cwd=BASE_DIR,
    )
    # 等待 ready 信号，最多 30 秒
    deadline = time.time() + 30
    while time.time() < deadline:
        line = _proc.stdout.readline()
        if not line:
            time.sleep(0.1); continue
        try:
            msg = json.loads(line.strip())
            if msg.get('ready'):
                _ready = True
                logger.info("TTS worker 就绪")
            else:
                logger.error(f"TTS worker 失败: {msg.get('error')}")
            return
        except json.JSONDecodeError:
            continue
    logger.error("TTS worker 启动超时")

def _ensure_worker():
    global _proc, _ready
    if _proc is None or _proc.poll() is not None:
        _ready = False
        _start_worker()

def synthesize(text: str, voice: str = 'Junhao') -> bytes:
    with _lock:
        _ensure_worker()
        if not _ready:
            raise RuntimeError("TTS worker 未就绪")
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        try:
            req = json.dumps({"text": text, "output": tmp.name,
                              "voice": voice}, ensure_ascii=False) + '\n'
            _proc.stdin.write(req); _proc.stdin.flush()
            # 等待响应，最多 120 秒（ONNX 每次 subprocess 有启动开销）
            deadline = time.time() + 120
            while time.time() < deadline:
                line = _proc.stdout.readline()
                if not line:
                    time.sleep(0.2); continue
                try:
                    resp = json.loads(line.strip())
                    if not resp.get('ok'):
                        raise RuntimeError(resp.get('error', '合成失败'))
                    with open(tmp.name, 'rb') as f:
                        return f.read()
                except json.JSONDecodeError:
                    continue
            raise TimeoutError("TTS 超时")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

def warmup():
    threading.Thread(target=_ensure_worker, daemon=True).start()
