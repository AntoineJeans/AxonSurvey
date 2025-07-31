# Axon Estimation Project

A comprehensive workflow for automated axon detection and quantification in neural tissue images using deep learning and computer vision techniques.

## Table of Contents

### 1. Overview

This project provides a complete pipeline for automated axon detection and quantification in neural tissue microscopy images. The workflow combines traditional computer vision techniques with deep learning approaches to accurately identify, trace, and quantify axonal structures in complex biological images.

**Key Features:**
- Automated axon detection using UNet-based segmentation
- Multi-strategy tracing algorithms (deterministic and deep learning-based)
- Comprehensive evaluation framework with multiple metrics
- Flexible data preparation and sampling strategies
- Reproducible experiment management system

**Applications:**
- Neuroscience research and brain mapping
- Quantitative analysis of neural connectivity
- Automated histological image analysis
- Large-scale neural tissue screening

### 2. Installation

**Prerequisites:**
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for deep learning components)
- 8GB+ RAM (16GB+ recommended for large datasets)

**Setup Instructions:**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd projet
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **GPU Setup (optional but recommended):**
   - Install CUDA Toolkit (version 11.0 or higher)
   - Install cuDNN library
   - Verify PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`

**Dependencies:**
The project relies on several key libraries including PyTorch, OpenCV, NumPy, scikit-image, and various scientific computing packages. See `requirements.txt` for the complete list.

### 3. Project Structure

```
projet/
├── src/                          # Main source code
│   ├── dataprep/                 # Data preparation and preprocessing
│   │   ├── DataReader.py         # Image and annotation loading
│   │   ├── DataSampler.py        # Sampling strategies for training
│   │   └── TracingChecker.py     # Quality control for traces
│   ├── NNs/                      # Neural network components
│   │   ├── Unet.py              # UNet architecture implementation
│   │   ├── training.py          # Training pipeline
│   │   └── inference.py         # Inference utilities
│   ├── tracers/                  # Axon tracing algorithms
│   │   ├── BaseTracer.py        # Abstract tracer interface
│   │   ├── DLTracer.py          # Deep learning-based tracing
│   │   └── DeterministicTracer.py # Traditional CV tracing
│   ├── evaluation/               # Evaluation and benchmarking
│   │   ├── Evaluator.py         # Main evaluation framework
│   │   └── TracerEval.py        # Tracer-specific evaluation
│   ├── inference/                # Inference pipeline
│   │   └── BaseEstimator.py     # Abstract estimator interface
│   ├── image_estimators/         # Image analysis estimators
│   │   ├── BaseFeatureExtractor.py # Feature extraction interface
│   │   └── TracerExtractor.py   # Tracer-based feature extraction
│   ├── experiments/              # Experiment management
│   │   └── InferencePipeline.py  # End-to-end inference workflow
│   └── utils/                    # Utility functions
│       ├── imageio.py           # Image I/O operations
│       └── viz.py               # Visualization tools
├── data/                         # Data storage directory
├── trained_models/               # Pre-trained model storage
├── docs/                         # Documentation (Sphinx-generated)
└── *.ipynb                      # Jupyter notebook examples
```

**Key Modules:**

- **`dataprep/`**: Handles all data preprocessing, including image loading, sampling strategies, and quality control
- **`NNs/`**: Contains the UNet implementation and training/inference pipelines
- **`tracers/`**: Implements various axon tracing algorithms with a common interface
- **`evaluation/`**: Provides comprehensive evaluation metrics and benchmarking tools
- **`inference/`**: Manages the end-to-end inference pipeline
- **`image_estimators/`**: Feature extraction and analysis components
- **`experiments/`**: Experiment configuration and management system

### 4. Experiments

[To be added later]

### 5. Usage Examples

Projects using this workflow should follow a similar series of steps:

1. **Data acquisition and structuring**
Images should be taken, then added to a repository structure as described in the file_structure_creation notebook. Each raw image can be manually separated in different ROIs, specific regions to seperate for quantification.

2. **Sampling**
The sampling notebook should have parameters set and be executed to sample from selected regions.
When sampling, some number of small fixed size image patches are taken. 
Currently, this sampling can only be done randomly, but future work could use informative features to guide that sampling process. This also may eventually lead to a slightly different workflow with two-fold sampling.

If models are to be trained in any way, create a train and a test dataset. If no training or fitting is required (only features are to be extracted from images) only a test set is necessary.

3. **Annotation**
Manually or semi-automatically create a segmentation mask from sampled images. In cases of axon quantification, this can be done by tracing axons with NeuronJ as showed in **Make a video**.

4. **Property definition**
In this project, we assume small image patches have measurable properties, that can be obtained from expert-annotations, and predicted with different models. These properties should be defined in code. This can be done in the form of a simple function that takes as input an image patch and outputs a property value, as defined in src.utils.traceProps.py.

5. **Extractor definition**
This can be done in parralel to the previous tasks. Specific extractors should be developped that have a chance at obtaining features from small image patches that are highly correlated with expert-annotations. This should be done with knowledge of the field and from previous works.

For example, for axon density quantification, we safely assume well-performing segmentation models create masks whose foreground density is highly correlated with human-expert axon tracing density. We also assume commonly used threshold algorithms create informative features for axon density.

It can also be useful to create simple baseline, with low expectations of performance, to validate minimal performance from other models.

6. **Training**
Externally train and fine tune models using training data. For segmentation projects where axon density is to be measured, this can be done quickly with the NNs module, using a Unet Architecture and some predefined training parameters. For other projects, we leave to the user the responsibility of creating models that can accurately segment their specific images. 

7. **Validation and inference**

First define a list of models to test quantification performance, then, run through the steps of the automatic Inference Pipeline. This pipeline is described in greater detail in the inference notebook, but it allows for measuring the performance of each selected model on each group that should be quantified. Then, the best model in each group is selected to predict the mean properties of each region in each group.
Finally, these results should be made available to the user, either in the form of empiric graph showing fixed point predictions for each region group (current approach) or using statistical estimators and variance of the real population property mean using samples and model predictions given some assumptions (to develop in the future).

Can be first run in debug mode for generating more graphs than necessary. It should be useful to view model predictions on test data in that pipeline to make sure expectations for what each model is doing are being met. For example, simple RMSE graphs and inference graphs don't handle abberant images and predictions outside of normal ranges. 