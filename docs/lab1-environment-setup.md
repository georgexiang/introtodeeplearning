# Lab 1 Environment Setup

This document summarizes the local environment setup for running the Lab 1 notebooks in this repository.

Lab 1 uses both TensorFlow and PyTorch notebooks:

- `lab1/TF_Part1_Intro.ipynb`
- `lab1/PT_Part1_Intro.ipynb`
- `lab1/TF_Part2_Music_Generation.ipynb`
- `lab1/PT_Part2_Music_Generation.ipynb`

The setup below creates a project-local Python virtual environment, installs the Python frameworks and notebook tooling, installs the ABC music conversion tools, registers a Jupyter kernel, and verifies that the environment can load the Lab 1 data.

## System Environment

The setup runs on Linux with Python 3.10:

```bash
python3 --version
```

Expected Python version:

```text
Python 3.10.12
```

The machine has an NVIDIA Tesla T4 GPU available:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

Example output:

```text
Tesla T4, 595.58.03
```

## Install System Packages

Install `pip`, virtual environment support, and the audio tools used by the music generation notebooks:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv abcmidi timidity
```

Verify the installed command-line tools:

```bash
python3 -m pip --version
command -v abc2midi
command -v timidity
```

Expected paths include:

```text
/usr/bin/abc2midi
/usr/bin/timidity
```

## Create the Virtual Environment

Create a local virtual environment in the repository root:

```bash
cd /home/azureuser/introtodeeplearning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

The environment lives at:

```text
/home/azureuser/introtodeeplearning/.venv
```

## Install Python Dependencies

Install the Lab 1 frameworks and support packages:

```bash
python -m pip install \
  "tensorflow==2.15.1" \
  torch \
  comet_ml \
  numpy \
  regex \
  tqdm \
  scipy \
  matplotlib \
  ipython \
  ipykernel \
  jupyter \
  opencv-python \
  h5py \
  gym \
  opik \
  openai \
  transformers \
  datasets \
  peft \
  lion-pytorch
```

The `mitdeeplearning` package imports helper modules from this repository. Install the local package in editable mode:

```bash
python -m pip install "setuptools<81"
python -m pip install -e . --no-deps --no-build-isolation
```

The older `setup.py` imports `pkg_resources`, so `setuptools<81` and `--no-build-isolation` keep the editable install compatible with this repository.

## Register the Jupyter Kernel

Register the virtual environment as a notebook kernel:

```bash
python -m ipykernel install \
  --user \
  --name introtodeeplearning-lab1 \
  --display-name "Python (.venv introtodeeplearning lab1)"
```

Verify the kernel registration:

```bash
jupyter kernelspec list | grep introtodeeplearning-lab1
```

Use this kernel when you open the Lab 1 notebooks:

```text
Python (.venv introtodeeplearning lab1)
```

## TensorFlow and PyTorch CUDA Notes

The environment installs both TensorFlow and PyTorch. TensorFlow can use CUDA through:

```bash
python -m pip install "tensorflow[and-cuda]==2.15.1"
```

This installs CUDA 12 runtime wheels under `site-packages/nvidia/`. PyTorch 2.11 installs CUDA 13 runtime wheels under the same namespace. Because both packages place libraries in shared `nvidia/*/lib` directories, installing TensorFlow CUDA extras after PyTorch can overwrite the NCCL library that PyTorch expects.

If PyTorch fails with an error like this:

```text
ImportError: libtorch_cuda.so: undefined symbol: ncclCommWindowDeregister
```

restore PyTorch's CUDA libraries by reinstalling PyTorch:

```bash
python -m pip install --force-reinstall torch==2.11.0
```

Do not globally add TensorFlow CUDA library paths to `.venv/bin/activate` or to the Jupyter kernel `LD_LIBRARY_PATH`. That can make TensorFlow detect GPU libraries, but it can also break PyTorch CUDA. Keep the shared environment stable by allowing the installed wheels to resolve their own bundled libraries.

## Final Verification

Run this verification from a fresh virtual environment activation:

```bash
cd /home/azureuser/introtodeeplearning
unset LD_LIBRARY_PATH
source .venv/bin/activate

python - <<'PY'
import os
import torch

print('python', os.sys.executable)
print('torch', torch.__version__)
print('torch_cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('torch_cuda_device', torch.cuda.get_device_name(0))

import mitdeeplearning as mdl
print('mitdeeplearning_import_after_torch', 'ok')
songs = mdl.lab1.load_training_data()
print('songs_loaded', len(songs))
PY

python - <<'PY'
import tensorflow as tf

print('tensorflow', tf.__version__)
print('tensorflow_import', 'ok')
print('tensorflow_gpu_count', len(tf.config.list_physical_devices('GPU')))
PY

command -v abc2midi
command -v timidity
jupyter kernelspec list | grep introtodeeplearning-lab1
```

Verified output from this environment:

```text
python /home/azureuser/introtodeeplearning/.venv/bin/python
torch 2.11.0+cu130
torch_cuda_available True
torch_cuda_device Tesla T4
mitdeeplearning_import_after_torch ok
Found 817 songs in text
songs_loaded 817

tensorflow 2.15.1
tensorflow_import ok
tensorflow_gpu_count 1

/usr/bin/abc2midi
/usr/bin/timidity
introtodeeplearning-lab1    /home/azureuser/.local/share/jupyter/kernels/introtodeeplearning-lab1
```

TensorFlow may print cuDNN, cuFFT, cuBLAS factory registration messages and a TensorRT warning during import. These messages do not block TensorFlow import or Lab 1 execution.

## Comet API Key

The music generation notebooks use `comet_ml` for experiment tracking. Install `comet_ml` as part of the environment, but set the API key yourself inside the notebook:

```python
COMET_API_KEY = "<your-api-key>"
```

The notebooks include assertions or comments that remind you to provide this value before running the full training loop.

## Daily Use

Activate the environment before running commands from a terminal:

```bash
cd /home/azureuser/introtodeeplearning
source .venv/bin/activate
```

When running notebooks in VS Code or Jupyter, select this kernel:

```text
Python (.venv introtodeeplearning lab1)
```