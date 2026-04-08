import subprocess
import shutil
import glob
import os
import time
import sys
import ctypes
from ctypes.util import find_library
import socket
import shlex
from pathlib import Path
from packaging import version
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn

COLOR_RED    = "\033[1;31m"
COLOR_GREEN  = "\033[1;32m"
COLOR_YELLOW = "\033[1;33m"
COLOR_PURPLE = "\033[1;35m"
COLOR_RESET  = "\033[0m"
GPFS_MAGIC   = 0x47504653

port_ssh       = 55555
line_width     = 100
hugging_subdir = "huggingface/datasets"

def echo_info(msg: str):
    print(f"[{COLOR_GREEN}INFO{COLOR_RESET}] {msg}")

def echo_warn(msg: str):
    print(f"[{COLOR_YELLOW}WARN{COLOR_RESET}] {msg}")

def echo_erro(msg: str):
    print(f"[{COLOR_RED}ERRO{COLOR_RESET}] {msg}")
    exit(0)

def echo_back(cmd: str, capture: bool = False, blocking=True, show=True):
    if show:
        print(f"[{COLOR_PURPLE}EXEC{COLOR_RESET}] {cmd}")
    parts = shlex.split(cmd)
    if len(parts) >= 2 and parts[0] == "cd":
        path = parts[1]
        os.chdir(path)
        return 0
    if capture:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    else:
        process = subprocess.Popen(cmd, shell=True)
        if blocking:
            process.communicate()
            return None
        else:
            return process

def echo_line(length: int = 100, symbol: str = "-", header: str = None):
    if header is None:
        print(f"[{COLOR_GREEN}INFO{COLOR_RESET}] " + symbol * length)
    else:
        header_len = len(header)
        if header_len >= length:
            length = header_len
        sym_len = length - header_len - 2
        left = sym_len // 2 
        right = sym_len - left 
        line_str = f"[{COLOR_GREEN}INFO{COLOR_RESET}] " + symbol * left + f" {COLOR_GREEN}{header}{COLOR_RESET} " + symbol * right
        print(line_str)

def progress_bar(elapsed, total, length=50):
    fraction = min(elapsed / total, 1)
    filled = int(length * fraction)
    bar = "█" * filled + "-" * (length - filled)
    sys.stdout.write(f"\r[{bar}] {elapsed:.1f} / {total}s")
    sys.stdout.flush()

def may_break(main_log, pattern):
    try:
        state = echo_back(f"bash -l -c 'grep \"{pattern}\" -r {main_log} | grep -v grep'", capture=True, show=False)
        if state is not None and len(state) > 0:
            return True
    except subprocess.CalledProcessError:
        state = ""
    return False

def await_ready(logs, error_pattern, ready_pattern, timeout=300):
    if isinstance(logs, str):
        logs = [logs]

    start = time.time()
    for log in logs:
        for _ in range(timeout):
            if Path(log).is_file():
                break
            time.sleep(0.5)
    
    with Progress(
        TextColumn("[green]Running..."),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("infinite", total=None)
        ready = 0
        for _ in range(timeout):
            for log in logs:
                if may_break(log, error_pattern):
                    print()
                    return 1
                if may_break(log, ready_pattern):
                    ready += 1
            if ready == len(logs):
                break
            time.sleep(3)
            progress.update(task, advance=1)
    print()
    return 0

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def get_local_hostname():
    return socket.gethostname()

class StatFs(ctypes.Structure):
    _fields_ = [
        ("f_type",    ctypes.c_ulong),
        ("f_bsize",   ctypes.c_ulong),
        ("f_blocks",  ctypes.c_ulong),
        ("f_bfree",   ctypes.c_ulong),
        ("f_bavail",  ctypes.c_ulong),
        ("f_files",   ctypes.c_ulong),
        ("f_ffree",   ctypes.c_ulong),
        ("f_fsid",    ctypes.c_ulong * 2),
        ("f_namelen", ctypes.c_ulong),
        ("f_frsize",  ctypes.c_ulong),
        ("f_flags",   ctypes.c_ulong),
        ("f_spare",   ctypes.c_ulong * 4)
    ]

def use_dfs(fd: int) -> bool:
    libc = ctypes.CDLL(find_library("c"), use_errno=True)
    statfs = libc.statfs
    statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(StatFs)]
    statfs.restype = ctypes.c_int
    try:
        path = os.readlink(f"/proc/self/fd/{fd}")
    except OSError as e:
        echo_erro(f"readlink failed: {e}")
        os.close(fd)
        raise

    fs_info = StatFs()
    ret = statfs(path.encode(), ctypes.byref(fs_info))
    if ret != 0:
        echo_erro("error")
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return fs_info.f_type == GPFS_MAGIC

def has_command(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def cuda_info():
    # output: ['NVIDIA L40', ...]
    arch_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    ).stdout.splitlines()
    nvcc_ver = "8.0"
    try:
        # output: CompletedProcess(args=['nvcc', '--version'], returncode=0, stdout='nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Wed_Nov_22_10:17:15_PST_2023\nCuda compilation tools, release 12.3, V12.3.107\nBuild cuda_12.3.r12.3/compiler.33567101_0\n', stderr='')
        nvcc_result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, check=True
        ).stdout.splitlines()
        nvcc_ver  = nvcc_result[3].split(",")[1].split()[1]
    except:
        pass
    cuda_arch = arch_result[0].split()[1]
    return cuda_arch, nvcc_ver

def is_version_ge(v1: str, v2: str) -> bool:
    """
    Check if version string v1 is greater than or equal to v2.
    Example: is_version_ge("12.10", "12.8") -> True
    """
    return version.parse(v1) >= version.parse(v2)

def update_env():
    cmd = "bash -l -c 'source ~/.bashrc && env'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        os.environ[key] = value
    
    os.environ["UV_LINK_MODE"] = "copy"

def venv_active(venv):
    if not Path(venv).expanduser().resolve().is_dir():
        echo_erro(f"uv {venv} has not been initialized!")
        exit(0)

    echo_info(f"source {venv}/bin/activate")
    cmd = f"bash -l -c '{venv}/bin/activate && env'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        os.environ[key] = value

def venv_remove(venv):
    if not Path(venv).expanduser().resolve().is_dir():
        echo_erro(f"uv {venv} has not been initialized!")
        return
    if Path(venv+"_old").expanduser().resolve().is_dir():
        echo_back(f"rm -rf {venv}_old")
    echo_back(f"mv {venv} {venv}_old")

def pull_repo(repo, dst, link):
    if not Path(f"{dst}/{repo}").is_dir() and not Path(dst + "/" + repo.lower()):
        echo_back(f"git clone {link}")
    echo_back(f"cd {dst}/{repo.lower()}")
    echo_back(f"git pull")

def list_pip_versions(package):
    res = echo_back(f"pip index versions {package}", capture=True)
    latest, versions, msg = None, set(), ""
    for line in res.splitlines():
        if line.startswith(f"Available versions:"):
            items = line.strip().split(":")[1]
            items = items.strip().split(",")
            msg = line
            for ver in items:
                ver = ver.strip()
                latest = ver if len(versions)==0 else latest
                versions.add(ver)
            break

    return latest, versions, msg

def get_model(model: str, search_roots=None) -> str:
    """
    Resolve a model name to an absolute model path.
    
    - If `model` is already a path: resolve & validate.
    - Otherwise search configured model locations.
    """
    p = Path(model).expanduser().resolve()
    if p.exists():
        p = p.resolve()
        if not (p / "config.json").exists() and (p / "v1.0").is_dir():
            p = p / "v1.0"
        return str(p)

    model_home = search_roots or []
    model_path = None
    for base in model_home:
        base_dir = Path(base).expanduser().resolve()
        if not base_dir.is_dir():
            continue

        # find equivalent: find ${path} -name model
        # glob "**/name" recursively
        found = list(base_dir.glob(f"**/{model}"))
        if not found:
            continue
        p = found[0].resolve()

        if not (p / "config.json").exists() and (p / "v1.0").is_dir():
            p = p / "v1.0"

        model_path = str(p)
        break
    if not model_path:
        echo_erro(f"Failed to locate {model}")
    return model_path

def finalize_ray(venv, username, sshport, workers, parallel=32):
    """
    Tear down an existing (possibly stale) Ray cluster across all worker nodes.

    This function SSHes into each worker machine and force-stops Ray, then
    removes temporary Ray-related directories to ensure a clean environment
    for the next startup.
    """
    echo_info(f"Tearing down a stale Ray cluster")
    for worker in workers:
        sshinfo = f"ssh -T -p {sshport} {username}@{worker[0]}"
        echo_back(f'{sshinfo} "{venv} NUMEXPR_MAX_THREADS={parallel} ray stop -f > /dev/null"')
        echo_back(f"{sshinfo} 'rm -rf /tmp/ray*'")
        echo_back(f"{sshinfo} 'rm -rf /tmp/tmp*'")

def prepare_ray(venv, username, sshport, workers, required, devcnt, port_ray, head=None, parallel=32):
    """
    Launch a Ray cluster if the required number of GPUs is available.

    This function checks whether a Ray cluster is already running; if so,
    it reuses the existing cluster when sufficient resources are detected.
    Otherwise, it finalizes any stale cluster and starts the Ray head node
    and worker nodes via SSH, ensuring the correct GPU count per node.
    """
    avail = len(workers) * devcnt
    if avail < required:
        return False

    ray_status = echo_back(f"ray status 2>/dev/null || true", capture=True)
    active = echo_back(f'echo \"{ray_status}\" | grep node_ | wc -l', capture=True, show=False)
    active = int(active) * devcnt

    if active >= required:
         echo_info(f"Ray cluster detected as running; restart skipped.")
         return True
    if active > 0:
        finalize_ray(venv, username, sshport, workers, parallel)
    
    head = workers[0][0] if head is None else head
    for i, worker in enumerate(workers):
        if worker[0] == head:
            workers[0], workers[i] = workers[i], workers[0]
            break

    for i, worker in enumerate(workers):
        if i * devcnt >= required:
            break
        sshinfo = f"ssh -T -p {sshport} {username}@{worker[0]}"
        if i == 0:
            echo_info(f"Starting the head node {worker} ...")
            echo_back(f'{sshinfo} "{venv} unset RAY_USE_MULTIPROCESSING_CPU_COUNT && RAY_DISABLE_DOCKER_CPU_WARNING=1 NUMEXPR_MAX_THREADS={parallel} ray start --head --port={port_ray} --num-gpus={devcnt} > /dev/null"')
        else:
            if i == 1:
                echo_info(f"Starting the worker nodes ...")
            echo_back(f'{sshinfo} "{venv} unset RAY_USE_MULTIPROCESSING_CPU_COUNT && RAY_DISABLE_DOCKER_CPU_WARNING=1 NUMEXPR_MAX_THREADS={parallel} ray start --address={workers[0][1]}:{port_ray} --num-gpus={devcnt} > /dev/null"')
        time.sleep(5)

if __name__ == "__main__":
    echo_info("This is an info message")
    echo_warn("This is a warning message")
    echo_line(50)
    echo_line(50, "*", "HEADER")
    ret = echo_back("echo 'Hello from shell'")
    print(ret)
    print(f"Hostname={get_local_hostname()}, IP={get_local_ip()}")
    print(f"nvidia-smi is installed!" if has_command("nvidia-smi") else "nvidia-smi is not installed!")
    print(
        is_version_ge("2.8", "12.8"),
        is_version_ge("11.10", "12.8"),
        is_version_ge("12.8", "12.8"),
        is_version_ge("12.9", "12.8"),
        is_version_ge("12.18", "12.8"),
        is_version_ge("13.8", "12.8")
    )

    update_env()
    print(os.environ["PATH"])
    list_pip_versions("vllm")
    list_pip_versions("sglang")
