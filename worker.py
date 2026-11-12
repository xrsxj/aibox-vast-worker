"""Vast.ai PyWorker for the AI box's cloud workers (run by Vast's start_server.sh from this repo's root).

AIBOX_ROLE=images   -> /draw forwarded to drawserver.py (ComfyUI, Qwen-Image 2.1 NVFP4) on 127.0.0.1:18300
AIBOX_ROLE=captions -> /v1/chat/completions forwarded to llama-server (FLUX 3 Klein PUP) on 127.0.0.1:18400
onstart.sh writes AIBOX_READY / AIBOX_FATAL into /var/log/aibox/server.log, which this worker tails.
"""
import copy
import json
import os
import random

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

ROLE = os.environ.get('AIBOX_ROLE', 'images')
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = '/var/log/aibox/server.log'
COST = 100.0  # one image or one caption; the box's client sends the same cost

if ROLE == 'images':
    BENCH = json.load(open(os.path.join(HERE, 'bench_workflow.json')))

    def bench():
        wf = copy.deepcopy(BENCH)
        wf['6']['inputs']['seed'] = random.randint(1, 2**31)
        return {'id': 'bench', 'workflow': wf, 'refs': {}}

    PORT, handler = 18300, HandlerConfig(
        route='/draw', allow_parallel_requests=True, max_queue_time=90.0,
        workload_calculator=lambda payload: COST,
        benchmark_config=BenchmarkConfig(generator=bench, runs=1, concurrency=2, do_warmup=True))
else:
    def bench():
        return {'model': 'flux3-pup', 'max_tokens': 200, 'temperature': 0.2,
                'messages': [{'role': 'user', 'content': 'A lighthouse keeper on wet rocks at dusk, waves behind her.'}]}

    PORT, handler = 18400, HandlerConfig(
        route='/v1/chat/completions', allow_parallel_requests=True, max_queue_time=60.0,
        workload_calculator=lambda payload: COST,
        benchmark_config=BenchmarkConfig(generator=bench, runs=1, concurrency=4, do_warmup=True))

Worker(WorkerConfig(
    model_server_url='http://127.0.0.1', model_server_port=PORT, model_log_file=LOG,
    model_healthcheck_url='/health', handlers=[handler],
    log_action_config=LogActionConfig(on_load=['AIBOX_READY'], on_error=['AIBOX_FATAL'], on_info=['AIBOX_INFO']),
)).run()
