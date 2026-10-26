#!/usr/bin/env python3
"""Install and run MAMMO-LENS on this PC. Requires Python 3.9+ and Internet for setup."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
UV_VERSION = '0.12.23'


def environment():
    env = os.environ.copy()
    # Always keep the project environment local to this checkout, even if the
    # launching terminal inherited a different project's uv configuration.
    env['UV_PROJECT_ENVIRONMENT'] = str(ROOT / '.venv')
    env.setdefault('UV_CACHE_DIR', str(ROOT / '.local-cache' / 'uv'))
    env.setdefault('TORCH_HOME', str(ROOT / 'state' / 'torch'))
    env.setdefault('MAMMO_ARTIFACTS', str(ROOT / 'artifacts'))
    env.setdefault('MAMMO_STATE', str(ROOT / 'state'))
    env.setdefault('MAMMO_MANIFEST', str(ROOT / 'data' / 'manifest.csv'))
    env['PYTHONUNBUFFERED'] = '1'
    return env


def uv_command(env):
    installed = shutil.which('uv')
    if installed:
        return [installed]
    bootstrap = ROOT / '.bootstrap'
    python = bootstrap / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        print('Creating a local installer environment…', flush=True)
        venv.EnvBuilder(with_pip=True).create(bootstrap)
    print('Installing uv from PyPI with TLS verification enabled…', flush=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--no-input', 'uv==' + UV_VERSION], cwd=ROOT, env=env, check=True)
    return [str(python), '-m', 'uv']


def open_when_ready(process, port, stopped):
    address = f'http://127.0.0.1:{port}'
    for _ in range(150):
        if stopped.wait(.2) or process.poll() is not None:
            return
        try:
            with urllib.request.urlopen(address + '/api/health', timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(address)
                    return
        except (OSError, urllib.error.URLError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--test', action='store_true', help='Run the Python suite after installation')
    args = parser.parse_args()
    if sys.version_info < (3, 9):
        parser.error('Install Python 3.9 or newer first; Python 3.12 is recommended')
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port between 1024 and 65535')
    if not (ROOT / 'uv.lock').is_file():
        parser.error('Extract the complete project folder before starting')
    env = environment()
    uv = uv_command(env)
    print('Preparing the locked Python 3.12 environment on this PC…', flush=True)
    subprocess.run(uv + ['sync', '--locked', '--all-extras', '--python', '3.12'],
                   cwd=ROOT, env=env, check=True)
    python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if args.test:
        subprocess.run([str(python), '-m', 'pytest', '-q'], cwd=ROOT, env=env, check=True)
    if args.setup_only:
        print('Local setup complete. Run this launcher again to open the application.')
        return
    print('Starting MAMMO-LENS locally. Keep this terminal open; press Ctrl+C to stop.', flush=True)
    stopped = threading.Event()
    process = subprocess.Popen([str(python), '-m', 'uvicorn', 'mammo_lens.api:app',
                                '--host', '127.0.0.1', '--port', str(args.port)], cwd=ROOT, env=env)
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(process, args.port, stopped), daemon=True).start()
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        stopped.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f'Local setup/start failed: {exc}', file=sys.stderr)
        sys.exit(1)
