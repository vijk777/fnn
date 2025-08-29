# Foundation model of neural activity predicts response to new stimulus types

[[`Paper`](https://www.nature.com/articles/s41586-025-08829-y)] [[`BibTeX`](#Citation)]

Eric Y Wang, Paul G Fahey, Zhuokun Ding, Stelios Papadopoulos, Kayla Ponder, Marissa A Weis, Andersen Chang, Taliah Muhammad, Saumil Patel, Zhiwei Ding, Dat Tran, Jiakun Fu, Casey M Schneider-Mizell, R Clay Reid, Forrest Collman, Nuno Maçarico da Costa, Katrin Franke, Alexander S Ecker, Jacob Reimer, Xaq Pitkow, Fabian H Sinz, Andreas S Tolias

# Model Overview

<p align="center">
<img src="./images/architecture.png" width="90%"></img>
</p>

Model of the visual cortex. The left panel (green) depicts an *in vivo* recording session of excitatory neurons from several areas (V1, LM, RL, AL) and layers (L2/3, L4, L5) of the mouse visual cortex. The right panel (blue) shows the architecture of the model and the flow of information from inputs (visual stimulus, eye position, locomotion, and pupil size) to outputs (neural activity). Underlined labels denote the four main modules of the model: perspective, modulation, core, and readout. For the modulation and core, the stacked planes represent feature maps. For the readout, the blue boxes represent the core’s output features at the readout position of the neuron, and the fanning black lines represent readout feature weights. The top of the schematic displays the neural activity for a sampled set of neurons. For two example neurons, *in vivo* and *in silico* responses are shown (green and blue, respectively).

NOTES: 
* The foundation model paper describes two types of recurrent network in the core: Conv-LSTM and CvT-LSTM. The core type released and supported in this package is CvT-LSTM.
* The model released and supported in this package is the digital twin of the MICrONS mouse. See tutorials for more detail.


# Demos
This repository contains demo Jupyter notebooks for:
1. Downloading the MICrONS digital twin
2. Using the MICrONS digital twin
3. Downloading the MICrONS digital twin computed properties
4. Preparing data for training a new digital twin (via transferring the frozen foundation core)
5. Training new digital twin
6. Evaluating new digital twin

To run the notebooks, launch the Jupyter Lab environment inside the Docker container. See below for access instructions.  

# Requirements

## Hardware

CPU-only support is available, but a modern GPU is recommended (tested on NVIDIA 3090, V100, A10, A100).

## Software

This package is written for Python (3.8+) and is supported on Linux (tested on Ubuntu 18.04, 20.04, 22.04, 24.04).

For launching the Docker container:
- Docker >= 20.10
- Docker Compose v2+

For GPU support:
- NVIDIA Container Toolkit

# Installation

There are two ways to install this repository. 

## 1. pip 

```bash
pip install git+https://github.com/cajal/fnn.git
```

##  2. Docker (RECOMMENDED)

### A. Clone this repository:

```bash
git clone https://github.com/cajal/fnn.git
```

### B. Navigate to the `deploy` directory

From the root directory of the repo, run:

```bash
cd deploy
```

### C. Create an `.env` file

The `.env` file can be empty, it just needs to exist at the same level as `docker-compose.yml`.

The following lines can be added to the `.env` file to customize the container (replace * with your own values):

**Jupyter:**
```
JUPYTER_TOKEN=*       # your desired password;          if omitted, there is no password prompt
JUPYTER_PORT_HOST=*   # your desired port on host;      if omitted, defaults to: 8888
```

**Image source & tag:**
```
IMAGE_REGISTRY=*      # your local registry;            if omitted, defaults to: ghcr.io
IMAGE_NAMESPACE=*     # desired namespace;              if omitted, defaults to: cajal
IMAGE_NAME=*          # desired image name;             if omitted, defaults to: foundation
IMAGE_TAG=*           # desired image tag (e.g. dev);   if omitted, defaults to: latest
```
### D. Launch the container.

By default, (i.e. if no environment variables are provided) the image tag will resolve to: `ghcr.io/cajal/foundation:latest`. 

When you start the service, Docker will pull the image only if it is not already present locally. To update to the newest version, run `docker compose pull` explicitly.

Run one of the following commands:

For GPU support:

```bash
docker compose up -d fnn
```

For CPU only:

```bash
docker compose up -d fnn-cpu
```

To explicitly pull the latest image:

```bash
docker compose pull fnn
```

### E. Access the container

Jupyter lab can be accessed at: `http://<host-ip>:<JUPYTER_PORT_HOST>/`, where `JUPYTER_PORT_HOST` defaults to `8888`.

If `JUPYTER_TOKEN` is set, use it to authenticate.

Once inside the Jupyter lab environment, you should be automatically directed to the demo notebooks.

# Citation

```bibtex
@article{wang2025foundation,
  title={Foundation model of neural activity predicts response to new stimulus types},
  author={Wang, Eric Y. and Fahey, Paul G. and Ding, Zhuokun and Papadopoulos, Stelios and Ponder, Kayla and Weis, Marissa A. and Chang, Andersen and Muhammad, Taliah and Patel, Saumil and Ding, Zhiwei and Tran, Dat and Fu, Jiakun and Schneider-Mizell, Casey M. and Reid, R. Clay and Collman, Forrest and da Costa, Nuno Ma{\c c}arico and Franke, Katrin and Ecker, Alexander S. and Reimer, Jacob and Pitkow, Xaq and Sinz, Fabian H. and Tolias, Andreas S.},
  journal={Nature},
  volume={640},
  number={8058},
  pages={470--477},
  year={2025},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41586-025-08829-y}
}
```
