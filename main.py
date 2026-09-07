import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
steps = [
    ('Prepare data', ['python', 'prepare_data.py']),
    ('Train forecasters', ['python', 'train_forecasters.py']),
    ('DDQN intervals', ['python', 'ddqn_interval.py'])
]

for title, cmd in steps:
    print('=' * 60)
    print(f'{title} -> {" ".join(cmd)}')
    print('=' * 60)
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        print(f'{title} failed with return code {result.returncode}')
        sys.exit(result.returncode)

print('All steps finished, check the generated npz and png files.')
