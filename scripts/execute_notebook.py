"""Execute the walkthrough using the current Python environment."""
import os
from pathlib import Path
import tempfile
import sys
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
notebook = root / 'notebooks' / 'portfolio_research.ipynb'
with tempfile.TemporaryDirectory() as directory:
    kernel = Path(directory) / 'kernels' / 'markowitz'
    kernel.mkdir(parents=True)
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Markowitz', 'language': 'python'}))
    os.environ['JUPYTER_PATH'] = directory + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    os.environ['IPYTHONDIR'] = str(Path(directory) / 'ipython')
    nb = nbformat.read(notebook, as_version=4)
    NotebookClient(nb, timeout=180, kernel_name='markowitz', resources={'metadata': {'path': str(root)}}).execute()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbformat.write(nb, notebook)
print('Notebook executed successfully.')
