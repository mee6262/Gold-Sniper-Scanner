import time
from datetime import datetime

from gold_sniper_v2 import run_once

INTERVAL_SECONDS = int(__import__('os').getenv('SCAN_INTERVAL_SECONDS', '60'))


def main():
    print('=' * 60)
    print('🔱 Gold-Sniper-Scanner V2 Hybrid Runner')
    print(f'⏱️ Poll interval: {INTERVAL_SECONDS}s')
    print('=' * 60)
    while True:
        started = datetime.now()
        print(f'\n🚀 Scan: {started:%Y-%m-%d %H:%M:%S}')
        try:
            run_once()
        except Exception as exc:
            print(f'❌ Runner error: {exc}')
        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    main()
