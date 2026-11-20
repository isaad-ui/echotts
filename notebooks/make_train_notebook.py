"""Generate notebooks/train_on_gpu.ipynb. Run: python notebooks/make_train_notebook.py

Kept as a script so the notebook stays reviewable in git diffs.
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
REPO = "https://github.com/isaad-ui/echotts.git"

cells = [
    md("# EchoTTS · train on a cloud GPU\n\n"
       "Works on **Google Colab** (Runtime → Change runtime type → GPU) and **Kaggle** (Accelerator → GPU, "
       "Internet on). Checkpoints go to persistent storage, so when the session is cut off you just run "
       "all cells again and training resumes from the newest checkpoint.\n\n"
       "Run cells top to bottom. Only run ONE of the two training cells at a time."),
    code("!nvidia-smi"),
    md("## 1. Persistent storage\nColab → Google Drive. Kaggle → `/kaggle/working` (save the notebook version "
       "with *Save output* to keep it, or upload checkpoints to a Kaggle dataset)."),
    code("import os, pathlib\n"
         "ON_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content')\n"
         "ON_KAGGLE = os.path.exists('/kaggle')\n"
         "if ON_KAGGLE:\n"
         "    PERSIST = pathlib.Path('/kaggle/working/echotts_store'); WORK = pathlib.Path('/kaggle/temp')\n"
         "else:\n"
         "    from google.colab import drive\n"
         "    drive.mount('/content/drive')\n"
         "    PERSIST = pathlib.Path('/content/drive/MyDrive/echotts'); WORK = pathlib.Path('/content')\n"
         "PERSIST.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)\n"
         "print('persistent storage:', PERSIST)"),
    md("## 2. Code"),
    code(f"REPO_DIR = WORK / 'echotts'\n"
         f"if REPO_DIR.exists():\n"
         f"    !git -C {{REPO_DIR}} pull\n"
         f"else:\n"
         f"    !git clone {REPO} {{REPO_DIR}}\n"
         f"%cd {{REPO_DIR}}\n"
         f"!pip install -q -r requirements.txt\n"
         f"!pytest -q"),
    md("## 3. LJSpeech\nOn Kaggle you can instead *Add data* → search **LJSpeech** and it is used automatically."),
    code("import glob\n"
         "attached = glob.glob('/kaggle/input/**/LJSpeech-1.1', recursive=True)\n"
         "if attached:\n"
         "    DATA = pathlib.Path(attached[0])\n"
         "else:\n"
         "    DATA = WORK / 'data' / 'LJSpeech-1.1'\n"
         "    if not (DATA / 'metadata.csv').exists():\n"
         "        (WORK / 'data').mkdir(exist_ok=True)\n"
         "        !wget -q -O {WORK}/data/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2\n"
         "        !tar -xjf {WORK}/data/LJSpeech-1.1.tar.bz2 -C {WORK}/data && rm {WORK}/data/LJSpeech-1.1.tar.bz2\n"
         "print(DATA, len(list((DATA / 'wavs').glob('*.wav'))), 'wavs')"),
    md("## 4. Precompute mels (once)\nMels are stored in persistent storage and linked into the data folder, "
       "so later sessions skip this step. Read-only inputs (Kaggle) get a writable copy of metadata + a link to the wavs."),
    code("import shutil\n"
         "MEL_STORE = PERSIST / 'mels'; MEL_STORE.mkdir(exist_ok=True)\n"
         "TRAIN_DATA = WORK / 'ljs_view'\n"
         "TRAIN_DATA.mkdir(exist_ok=True)\n"
         "for name, target in [('wavs', DATA / 'wavs'), ('mels', MEL_STORE)]:\n"
         "    link = TRAIN_DATA / name\n"
         "    if not link.exists(): link.symlink_to(target, target_is_directory=True)\n"
         "shutil.copy(DATA / 'metadata.csv', TRAIN_DATA / 'metadata.csv')\n"
         "!python -m echotts.precompute_mels --data_dir {TRAIN_DATA}\n"
         "print(len(list(MEL_STORE.glob('*.npy'))), 'mels ready')"),
    md("## 5. TensorBoard\nWatch `train/total` fall in the first 500 steps and check the alignment images every few thousand steps."),
    code("%load_ext tensorboard\n%tensorboard --logdir {PERSIST}/runs"),
    md("## 6a. Train the acoustic model\nResumes automatically. Add `--batch_size 16` if you run out of GPU memory, "
       "`--amp` for mixed precision (faster on T4/A100)."),
    code("!python -m echotts.train_acoustic --data_dir {TRAIN_DATA} \\\n"
         "    --checkpoint_dir {PERSIST}/checkpoints/acoustic --log_dir {PERSIST}/runs/acoustic"),
    md("## 6b. Train the vocoder\nTrains on real recordings, independent of the acoustic model, so it can run in a "
       "separate session (or a second free GPU account) in parallel."),
    code("!python -m echotts.train_vocoder --data_dir {TRAIN_DATA} \\\n"
         "    --checkpoint_dir {PERSIST}/checkpoints/vocoder --log_dir {PERSIST}/runs/vocoder"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
nbf.write(nb, Path(__file__).with_name("train_on_gpu.ipynb"))
print("wrote train_on_gpu.ipynb")
