import hashlib
import json
from pathlib import Path
import subprocess


def command(*args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


paths = ['engine/pinfer_run_fast', 'tmp/gpu-engine/pinfer_run_gpu8',
         'tmp/gpu-engine/lib-fallback/libpinfer.so']
memory = {}
for line in Path('/proc/meminfo').read_text().splitlines():
    key, value = line.split(':', 1)
    if key in ('MemTotal', 'MemAvailable', 'Shmem', 'CmaTotal', 'CmaFree'):
        memory[key + '_kib'] = int(value.split()[0])
kernel = subprocess.run(['dmesg'], text=True, capture_output=True, check=True).stdout
gpu_messages = [s for s in kernel.splitlines() if 'v3d' in s.lower() or
                ('drm' in s.lower() and ('reset' in s.lower() or 'timed out' in s.lower()))]
print('RESULT ' + json.dumps({
    'gpu_health': True, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'uptime_s': float(Path('/proc/uptime').read_text().split()[0]),
    'throttled': command('vcgencmd', 'get_throttled'),
    'cpu_clock': command('vcgencmd', 'measure_clock', 'arm'),
    'v3d_clock': command('vcgencmd', 'measure_clock', 'v3d'),
    'memory': memory, 'gpu_kernel_messages': gpu_messages[-20:],
    'md5': {p: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in paths},
    'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
}), flush=True)
