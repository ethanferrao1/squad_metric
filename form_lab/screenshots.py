"""Screenshot every tab of the app, wide and narrow, offline.

    python -m form_lab.screenshots [team_id]

Starts the app with FPL_OFFLINE=1, drives the installed Microsoft Edge with
Playwright, and saves PNGs to form_lab/results/screens/.
"""
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'form_lab' / 'results' / 'screens'
PORT = 8599
TABS = ['My Team', 'Lineup', 'Transfers', 'Players', 'News', 'Build']
SIZES = {'wide': (1440, 2400), 'narrow': (420, 3600)}


def start_app():
    env = {**os.environ, 'FPL_OFFLINE': '1'}
    proc = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.headless',
         'true', '--server.port', str(PORT)], cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f'http://localhost:{PORT}/_stcore/health',
                                        timeout=2) as r:
                if r.read() == b'ok':
                    return proc
        except OSError:
            time.sleep(1)
    proc.kill()
    raise RuntimeError('the app did not start')


def shoot(page, name):
    page.wait_for_timeout(2500)
    page.screenshot(path=str(OUT / f'{name}.png'))
    print(f'  {name}.png', flush=True)


def main(team='3265946'):
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    proc = start_app()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel='msedge', headless=True)
            for size, (w, h) in SIZES.items():
                page = browser.new_page(viewport={'width': w, 'height': h})
                page.goto(f'http://localhost:{PORT}/?team={team}')
                page.get_by_text('Team rating').first.wait_for(timeout=300_000)
                bar = page.locator('.st-key-section')
                for i, tab in enumerate(TABS, 1):
                    bar.get_by_text(tab).click()
                    page.wait_for_timeout(1500)
                    page.locator('[data-testid="stSpinner"], .sm-spin').first.wait_for(
                        state='detached', timeout=120_000)
                    shoot(page, f'{size}_{i}_{tab.lower().replace(" ", "_")}')
                if size == 'wide':
                    bar.get_by_text('My Team').click()
                    page.get_by_text('Team rating').first.wait_for(timeout=120_000)
                    page.locator('button:visible').filter(
                        has_text=re.compile(r'^Haaland$')).first.click()
                    page.get_by_role('dialog').wait_for(timeout=60_000)
                    page.wait_for_timeout(4000)
                    shoot(page, 'wide_7_player_detail')
                page.close()
            browser.close()
    finally:
        proc.kill()


if __name__ == '__main__':
    main(*sys.argv[1:])
