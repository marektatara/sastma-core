# About

The repository is an experimental code for segmentation-aided stereo matching method with confidence refinement. This method is based on the paper "Segmentation-aided stereo matching with confidence refinement" by Marek S. Tatara, Jan Glinko, Michał Czubenko, Marek Grzegorek, Zdzisław Kowalczuk. The paper is available at ...

# Getting Started

## Prerequisites

The code is written in Python 3.12.3. Poetry is used for managing dependencies. Install dependencies by running the following command:

```bash
poetry install
```

## Usage

The code is currently oriented around notebooks. There are the following notebooks available:
- `sastma.ipynb` - the main notebook with the implementation of the method
- `pixel-matching.ipynb` - the notebook with the implementation of the pixel-level matching method
- `confidence-refinement.ipynb` - the notebook with the implementation of the confidence refinement method

