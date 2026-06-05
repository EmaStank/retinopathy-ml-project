# Diabetic Retinopathy ML

Machine learning project for detecting and classifying diabetic retinopathy from fundus images.

## Project Structure

```
diabetic_retinopathy_ml/
│
├── data/              # Dataset directory
├── models/            # Trained model checkpoints
├── outputs/           # Results and predictions
├── download_data.py   # Download and prepare dataset
├── train_models.py    # Train ML models
├── evaluate_model.py  # Evaluate model performance
└── README.md          # This file
```

## Getting Started

1. Activate the virtual environment:
   ```powershell
   & ".\.venv\Scripts\Activate.ps1"
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Download data: `python download_data.py`
4. Train models: `python train_models.py`
5. Evaluate: `python evaluate_model.py`

> If you run `python load_data.py` without the venv activated, Windows may use your system Python and fail to import `scipy`.

## Requirements

- Python 3.x
- `scipy`
- `pandas`
- TensorFlow/PyTorch
- NumPy, scikit-learn

## License

MIT License
