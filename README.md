# SCG Machine Learning - Cardiac Analytics

A machine learning project for cardiac signal analysis and classification using HVDNet neural networks and signal processing techniques.

## Project Overview

This project implements deep learning models for analyzing patient cardiac data, including:
- **Task I**: Aortic Stenosis classification
- **Task II**: Multi-condition AS classification  
- **Task III**: Valve disease classification

Includes data visualization and signal processing tools via PyQt6 GUI and Streamlit web interface.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Setup Repository
```bash
git clone https://github.com/wetwafflws/Seismocardiography-Machine-Learning 
cd SCG_MachineLearning
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch (`torch`, `torchvision`, `torchaudio`)
- scikit-learn
- pandas
- numpy
- scipy
- PyQt5 (for inference-only desktop app)
- PyQt6 (for GUI)
- Streamlit (for web interface)
- pyqtgraph (for plotting)
- plotly (for interactive visualizations)

## Project Structure

```
SCG_MachineLearning/
├── machinelearning.py          # Main ML pipeline and GUI application
├── pyqt5_hvdnet_inference.py   # Inference-only PyQt5 app (no training)
├── TA_SVMD.py                  # Streamlit data viewer and signal processing
├── hvdnet_task_i_1.pt          # Pre-trained model for Task I
├── hvdnet_task_ii_1.pt         # Pre-trained model for Task II
├── hvdnet_task_iii_1.pt        # Pre-trained model for Task III
├── Data/                        # Patient data (CSV files) - NOT included in repo*
│   ├── Cleaned_CP-01.csv
│   ├── Cleaned_UP-01.csv
│   └── ...
└── Saved_Peaks/                # Output directory - NOT included in repo*
    └── (generated peak data)

*To be added when running locally
```

## Data Setup

The data is not included in the repository and can be found through this study:
Yang C, Fan F, Aranoff N, Green P, Li Y, Liu C and Tavassolian N (2021) An Open-Access Database for the Evaluation of Cardio-Mechanical Signals From Patients With Valvular Heart Diseases. Front. Physiol. 12:750221. doi: 10.3389/fphys.2021.750221

### Adding Your Data

1. Create a `Data/` folder in the project root directory:
   ```bash
   mkdir Data
   ```

2. Place your cleaned CSV patient files in this folder:
   ```
   Data/
   ├── Cleaned_CP-01.csv
   ├── Cleaned_CP-02.csv
   ├── Cleaned_UP-01.csv
   └── ...
   ```

### File Path Convention

All Python scripts use **relative paths** to access the Data folder:

```python
# Correct - relative to project root
data_dir = "Data"
HVDNetDataLoader(data_dir="Data")

# Data files are accessed as:
# Data/Cleaned_CP-01.csv
# Data/Cleaned_UP-01.csv
```

## Usage

### Option 1: PyQt6 GUI Application
```bash
python machinelearning.py
```

### Option 2: PyQt5 Inference-Only Application
```bash
python pyqt5_hvdnet_inference.py
```

This app only performs:
- model checkpoint loading
- feedforward inference
- signal and attention visualization
- single-patient and batch inference export

### Option 3: Streamlit Web Interface
```bash
streamlit run TA_SVMD.py
```

## Pre-trained Models

Three pre-trained PyTorch models are included:
- `hvdnet_task_i_1.pt` - Task I model weights
- `hvdnet_task_ii_1.pt` - Task II model weights
- `hvdnet_task_iii_1.pt` - Task III model weights

These are automatically loaded when running inference.

## Notes

- Data sampling is standardized to 256 Hz in the pipeline
- The project handles both UP- and CP- patient database prefixes
- Original sampling frequencies are auto-detected based on patient ID
- Signal processing includes resampling, filtering, and decomposition

