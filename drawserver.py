"""One image per call, drawn by the ComfyUI beside it (127.0.0.1:18188). The Vast PyWorker forwards /draw here.

POST /draw  {"id": "...", "workflow": {ComfyUI API graph}, "refs": {"name.png": "<base64>"}}
  -> {"id", "prompt_id", "status": "success"|"error", "messages": [...ComfyUI status messages...],
      "outputs": {node: {"images": [{"filename", "subfolder", "type", "b64"}]}}, "error": "..."}
GET /health -> 200 once ComfyUI answers.

Prints AIBOX_READY to its log (the PyWorker's on_load line) once ComfyUI answers, AIBOX_FATAL if it never does.
"""
import asyncio
import base64
import json
import os
import time

from aiohttp import ClientSession, ClientTimeout, web

COMFY = os.environ.get('AIBOX_COMFY', 'http://127.0.0.1:18188')
ROOT = os.environ.get('AIBOX_COMFY_DIR', '/workspace/ComfyUI')
INPUT, OUTPUT = os.path.join(ROOT, 'input'), os.path.join(ROOT, 'output')
PORT = int(os.environ.get('AIBOX_DRAW_PORT', '18300'))
TIMEOUT_S = 900
http: ClientSession = None


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def _safe(base, *parts):
    path = os.path.realpath(os.path.join(base, *parts))
    if not path.startswith(os.path.realpath(base) + os.sep):
        raise ValueError(f'path outside {base}: {parts}')
    return path


async def draw(req):
    body = await req.json()
    rid, wf, refs = body.get('id') or '', body.get('workflow'), body.get('refs') or {}
    if not isinstance(wf, dict):
        return web.json_response({'id': rid, 'status': 'error', 'error': 'no workflow'}, status=400)
    for name, b64 in refs.items():
        path = _safe(INPUT, name)
        if not os.path.exists(path):
            data = base64.b64decode(b64)
            tmp = path + '.part'
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, path)
    t0 = time.time()
    async with http.post(f'{COMFY}/prompt', json={'prompt': wf, 'client_id': 'aibox-worker'}) as r:
        answer = await r.json(content_type=None)
    pid = answer.get('prompt_id')
    if r.status != 200 or not pid:
        log(f'{rid}: ComfyUI refused it: {json.dumps(answer)[:300]}')
        return web.json_response({'id': rid, 'status': 'error', 'error': json.dumps(answer)[:2000],
                                  'refused': True})
    hist = None
    while time.time() - t0 < TIMEOUT_S:
        await asyncio.sleep(0.25)
        async with http.get(f'{COMFY}/history/{pid}') as r:
            h = await r.json(content_type=None)
        if pid in h and (h[pid].get('status') or {}).get('completed') is not None and \
                ((h[pid].get('status') or {}).get('status_str') in ('success', 'error')):
            hist = h[pid]
            break
    if hist is None:
        async with http.post(f'{COMFY}/interrupt', json={}) as r:
            pass
        return web.json_response({'id': rid, 'prompt_id': pid, 'status': 'error', 'error': 'timed out'})
    st = hist.get('status') or {}
    out = {}
    for node, o in (hist.get('outputs') or {}).items():
        ims = []
        for im in o.get('images') or []:
            if im.get('type') != 'output':
                continue
            path = _safe(OUTPUT, im.get('subfolder') or '', im['filename'])
            with open(path, 'rb') as f:
                data = f.read()
            os.remove(path)
            ims.append({**im, 'b64': base64.b64encode(data).decode()})
        if ims:
            out[node] = {'images': ims}
    async with http.post(f'{COMFY}/history', json={'delete': [pid]}) as r:
        pass
    log(f'{rid}: {st.get("status_str")} in {time.time() - t0:.1f}s')
    return web.json_response({'id': rid, 'prompt_id': pid, 'status': st.get('status_str'),
                              'messages': st.get('messages') or [], 'outputs': out})


async def health(req):
    try:
        async with http.get(f'{COMFY}/system_stats', timeout=ClientTimeout(total=5)) as r:
            return web.json_response({'ok': r.status == 200}, status=200 if r.status == 200 else 503)
    except Exception as e:
        return web.json_response({'ok': False, 'error': str(e)[:200]}, status=503)


async def startup(app):
    global http
    http = ClientSession(timeout=ClientTimeout(total=TIMEOUT_S + 60))
    os.makedirs(INPUT, exist_ok=True)

    async def wait_ready():
        for _ in range(1800):  # ComfyUI starts after the model download
            try:
                async with http.get(f'{COMFY}/system_stats', timeout=ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        log('AIBOX_READY ComfyUI answers')
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
        log('AIBOX_FATAL ComfyUI never answered')
    asyncio.get_running_loop().create_task(wait_ready())


app = web.Application(client_max_size=256 * 1024 * 1024)
app.router.add_post('/draw', draw)
app.router.add_get('/health', health)
app.on_startup.append(startup)

if __name__ == '__main__':
    web.run_app(app, host='127.0.0.1', port=PORT, access_log=None, print=None)
