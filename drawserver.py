"""One image per call, drawn by a ComfyUI beside it: one ComfyUI per GPU (127.0.0.1:18188, 18189, ...; AIBOX_GPUS of
them), each image to the least busy one. The Vast PyWorker forwards /draw here.

POST /draw  {"id": "...", "workflow": {ComfyUI API graph}, "refs": {"name.png": "<base64>"}}
  -> {"id", "prompt_id", "status": "success"|"error", "messages": [...ComfyUI status messages...],
      "outputs": {node: {"images": [{"filename", "subfolder", "type", "b64"}]}}, "error": "...", "gpu": n}
GET /health -> 200 once a ComfyUI answers.

Identity check at start: every GPU draws bench_workflow.json at IDENTITY_SEED and must match the AI box's own 5070 Ti
pixel for pixel; a GPU that doesn't is left out. AIBOX_READY (the PyWorker's on_load line) once at least one passed,
AIBOX_FATAL if none did (Vast then replaces the worker).
"""
import asyncio
import base64
import json
import os
import time

from aiohttp import ClientSession, ClientTimeout, web

ROOT = os.environ.get('AIBOX_COMFY_DIR', '/workspace/ComfyUI')
INPUT = os.path.join(ROOT, 'input')
GPUS = int(os.environ.get('AIBOX_GPUS', '1'))
BASE_PORT = int(os.environ.get('AIBOX_COMFY_PORT', '18188'))
PORT = int(os.environ.get('AIBOX_DRAW_PORT', '18300'))
TIMEOUT_S = 900
HERE = os.path.dirname(os.path.abspath(__file__))
# bench_workflow.json at this seed, drawn on the AI box's own RTX 5070 Ti (driver 595.99, torch 2.15 cu134): sha256 of the
# decoded RGB pixels. A GPU that doesn't draw exactly this (other card model, other libraries) doesn't serve.
IDENTITY_SEED = 1234567
IDENTITY_SHA = os.environ.get('AIBOX_IDENTITY_SHA', '37bd930e52a825befa42bfc38f77ef9ac3a509e80b9ebbffa3a5c4de0ec0607b')
# one per GPU: its URL and output folder (onstart.sh starts ComfyUI i with --output-directory ROOT/output/gpu<i> when there
# are several), images in flight, and whether it passed the identity check
BACKENDS = [{'gpu': i, 'url': f'http://127.0.0.1:{BASE_PORT + i}',
             'output': os.path.join(ROOT, 'output', f'gpu{i}') if GPUS > 1 else os.path.join(ROOT, 'output'),
             'busy': 0, 'ok': False} for i in range(GPUS)]
http: ClientSession = None


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def _safe(base, *parts):
    path = os.path.realpath(os.path.join(base, *parts))
    if not path.startswith(os.path.realpath(base) + os.sep):
        raise ValueError(f'path outside {base}: {parts}')
    return path


def pixel_hash(png):
    import hashlib
    import io
    import numpy as np
    from PIL import Image
    return hashlib.sha256(np.asarray(Image.open(io.BytesIO(png)).convert('RGB')).tobytes()).hexdigest()


async def run(b, rid, wf, refs):
    """Draw one workflow on backend b -> the answer dict."""
    for name, b64 in refs.items():
        path = _safe(INPUT, name)
        if not os.path.exists(path):
            tmp = f'{path}.{os.getpid()}.{id(refs)}.part'
            with open(tmp, 'wb') as f:
                f.write(base64.b64decode(b64))
            os.replace(tmp, path)
    t0 = time.time()
    b['busy'] += 1
    try:
        async with http.post(f"{b['url']}/prompt", json={'prompt': wf, 'client_id': 'aibox-worker'}) as r:
            answer = await r.json(content_type=None)
        pid = answer.get('prompt_id')
        if r.status != 200 or not pid:
            log(f'{rid}: ComfyUI {b["gpu"]} refused it: {json.dumps(answer)[:300]}')
            return {'id': rid, 'status': 'error', 'error': json.dumps(answer)[:2000], 'refused': True, 'gpu': b['gpu']}
        hist = None
        while time.time() - t0 < TIMEOUT_S:
            await asyncio.sleep(0.25)
            async with http.get(f"{b['url']}/history/{pid}") as r:
                h = await r.json(content_type=None)
            st = (h.get(pid) or {}).get('status') or {}
            if st.get('completed') is not None and st.get('status_str') in ('success', 'error'):
                hist = h[pid]
                break
        if hist is None:
            async with http.post(f"{b['url']}/interrupt", json={}) as r:
                pass
            return {'id': rid, 'prompt_id': pid, 'status': 'error', 'error': 'timed out', 'gpu': b['gpu']}
        st = hist.get('status') or {}
        out = {}
        for node, o in (hist.get('outputs') or {}).items():
            ims = []
            for im in o.get('images') or []:
                if im.get('type') != 'output':
                    continue
                path = _safe(b['output'], im.get('subfolder') or '', im['filename'])
                with open(path, 'rb') as f:
                    data = f.read()
                os.remove(path)
                ims.append({**im, 'b64': base64.b64encode(data).decode()})
            if ims:
                out[node] = {'images': ims}
        async with http.post(f"{b['url']}/history", json={'delete': [pid]}) as r:
            pass
        log(f'{rid}: {st.get("status_str")} in {time.time() - t0:.1f}s on GPU {b["gpu"]}')
        return {'id': rid, 'prompt_id': pid, 'status': st.get('status_str'), 'messages': st.get('messages') or [],
                'outputs': out, 'gpu': b['gpu']}
    finally:
        b['busy'] -= 1


async def draw(req):
    body = await req.json()
    rid, wf, refs = body.get('id') or '', body.get('workflow'), body.get('refs') or {}
    if not isinstance(wf, dict):
        return web.json_response({'id': rid, 'status': 'error', 'error': 'no workflow'}, status=400)
    ok = [b for b in BACKENDS if b['ok']]
    if not ok:
        return web.json_response({'id': rid, 'status': 'error', 'error': 'no GPU ready'}, status=503)
    b = min(ok, key=lambda x: x['busy'])
    return web.json_response(await run(b, rid, wf, refs))


async def health(req):
    return web.json_response({'ok': any(b['ok'] for b in BACKENDS), 'gpus': [b['ok'] for b in BACKENDS]},
                             status=200 if any(b['ok'] for b in BACKENDS) else 503)


async def startup(app):
    global http
    http = ClientSession(timeout=ClientTimeout(total=TIMEOUT_S + 60))
    os.makedirs(INPUT, exist_ok=True)
    asyncio.get_running_loop().create_task(wait_ready())


async def check(b):
    """Wait for backend b's ComfyUI, then its identity image; b['ok'] if it matches the AI box pixel for pixel."""
    for _ in range(1800):  # ComfyUI starts after the model download
        try:
            async with http.get(f"{b['url']}/system_stats", timeout=ClientTimeout(total=5)) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        await asyncio.sleep(1)
    else:
        log(f'GPU {b["gpu"]}: ComfyUI never answered')
        return
    try:
        wf = json.load(open(os.path.join(HERE, 'bench_workflow.json')))
        wf['6']['inputs']['seed'] = IDENTITY_SEED
        out = await run(b, 'identity', wf, {})
        im = next(i for o in out['outputs'].values() for i in o['images'])
        got = await asyncio.to_thread(pixel_hash, base64.b64decode(im['b64']))
    except Exception as e:
        log(f'GPU {b["gpu"]}: identity check failed to run: {e.__class__.__name__}: {str(e)[:200]}')
        return
    if got != IDENTITY_SHA:
        log(f'GPU {b["gpu"]}: output differs from the AI box (pixels {got[:16]}, expected {IDENTITY_SHA[:16]}); left out')
        return
    b['ok'] = True
    log(f'GPU {b["gpu"]}: identity check passed (pixel-identical to the AI box)')


async def wait_ready():
    await asyncio.gather(*(check(b) for b in BACKENDS))
    n = sum(b['ok'] for b in BACKENDS)
    if n:
        log(f'AIBOX_READY {n} of {len(BACKENDS)} GPU(s) pixel-identical to the AI box')
    else:
        log('AIBOX_FATAL no GPU drew the identity image like the AI box')


app = web.Application(client_max_size=256 * 1024 * 1024)
app.router.add_post('/draw', draw)
app.router.add_get('/health', health)
app.on_startup.append(startup)

if __name__ == '__main__':
    web.run_app(app, host='127.0.0.1', port=PORT, access_log=None, print=None)
