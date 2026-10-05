import os
import sys
from pathlib import Path
root = str(Path(__file__).resolve().parents[1] / 'src')
sys.path.insert(0, root)
os.environ['PYTHONPATH'] = root + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
