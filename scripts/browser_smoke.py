#!/usr/bin/env python3
"""Headless Chromium smoke test against a running local MAMMO-LENS server."""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from websockets.sync.client import connect

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:8000/')
parser.add_argument('--output',default='artifacts/browser-smoke')
args=parser.parse_args()
from urllib.parse import urlsplit
if urlsplit(args.url).hostname not in ('127.0.0.1','localhost'):
    parser.error('This smoke runner only sends fixtures to a local server')
output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
chromium=shutil.which('chromium') or shutil.which('google-chrome')
if not chromium: parser.error('Install Chromium or Google Chrome first')
with socket.socket() as port_socket:
    port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1]
with tempfile.TemporaryDirectory(prefix='mammo-browser-') as profile:
    env=os.environ.copy();env['XDG_CACHE_HOME']=profile+'/cache';env['GSETTINGS_BACKEND']='memory'
    process=subprocess.Popen([chromium,'--headless','--no-sandbox','--disable-dev-shm-usage','--disable-gpu',
        '--no-first-run','--no-default-browser-check','--disable-background-networking',
        '--user-data-dir='+profile+'/profile','--remote-debugging-port='+str(port),'about:blank'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=1): break
            except OSError: time.sleep(.1)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json') as response:
            tab=json.load(response)[0]
        with connect(tab['webSocketDebuggerUrl'], proxy=None, open_timeout=10) as ws:
            counter=0
            def call(method,params=None):
                global counter
                counter+=1
                ws.send(json.dumps(dict(id=counter,method=method,params=params or {})))
                while True:
                    result=json.loads(ws.recv(timeout=20))
                    if result.get('id')==counter:
                        if 'error' in result: raise RuntimeError(result['error'])
                        return result['result']
            def evaluate(expression):
                result=call('Runtime.evaluate',{'expression':expression,'returnByValue':True})
                if 'exceptionDetails' in result: raise RuntimeError(result['exceptionDetails'])
                return result['result'].get('value')
            call('Page.enable')
            call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1300,deviceScaleFactor=1,mobile=False))
            call('Page.navigate',{'url':args.url})
            for _ in range(40):
                if evaluate('document.body?.dataset.ready')=='true': break
                time.sleep(.1)
            assert evaluate('document.body.dataset.ready')=='true'
            assert evaluate("document.querySelectorAll('#patches canvas').length")==4
            assert evaluate("document.getElementById('change').textContent")=='0.0000'
            evaluate("document.getElementById('noise').value=.1;document.getElementById('noise').dispatchEvent(new Event('input'))")
            assert float(evaluate("document.getElementById('change').textContent"))>0
            assert evaluate('latestReport.trained_inference_performed') is False
            evaluate("document.getElementById('view-report').click()")
            assert evaluate("document.getElementById('report-dialog').open") is True
            assert 'mean_absolute_pixel_change' in evaluate("document.getElementById('report-json').textContent")
            evaluate("document.getElementById('close-report').click();document.getElementById('reset').click()")
            assert evaluate("document.getElementById('change').textContent")=='0.0000'
            # Test a real browser file-upload event without transmitting the image anywhere.
            expression="""new Promise(resolve=>{const src=document.getElementById('reference');src.toBlob(blob=>{const dt=new DataTransfer();dt.items.add(new File([blob],'smoke-test.png',{type:'image/png'}));const input=document.getElementById('upload');input.files=dt.files;input.dispatchEvent(new Event('change'));resolve(true);},'image/png');})"""
            call('Runtime.evaluate',{'expression':expression,'awaitPromise':True})
            for _ in range(40):
                if evaluate('sourceType')=='user_uploaded_image': break
                time.sleep(.1)
            assert evaluate('sourceType')=='user_uploaded_image'
            assert evaluate('sourceName')=='smoke-test.png'
            evaluate("document.getElementById('analyze').click()")
            for _ in range(100):
                if 'Server analysis complete' in evaluate("document.getElementById('api-status').textContent"): break
                time.sleep(.1)
            assert 'Server analysis complete' in evaluate("document.getElementById('api-status').textContent")
            assert evaluate('latestReport.prediction') is None
            assert evaluate('latestReport.preprocessing')=='percentile_1_99.5_gray_256'
            evaluate("document.getElementById('noise').value=.05;document.getElementById('noise').dispatchEvent(new Event('input'))")
            assert evaluate('serverResponse') is None
            assert 'No trained model' in evaluate("document.getElementById('prediction-panel').textContent")
            evaluate("document.getElementById('audit-data').click()")
            for _ in range(30):
                if 'authentication' in evaluate("document.getElementById('training-notice').textContent"): break
                time.sleep(.1)
            assert 'authentication' in evaluate("document.getElementById('training-notice').textContent")
            evaluate("document.getElementById('phantom').click()")
            assert evaluate('sourceType')=='illustrative_phantom'
            # Desktop screenshot, then verify narrow layout has no horizontal overflow.
            image=call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True})
            (output/'desktop.png').write_bytes(base64.b64decode(image['data']))
            call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
            assert evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            mobile=call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True})
            (output/'mobile.png').write_bytes(base64.b64decode(mobile['data']))
            print('PASS: live frontend controls, upload, Python API analysis, stale-result invalidation, operator authentication, report dialog, and mobile layout.')
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill();process.wait()
