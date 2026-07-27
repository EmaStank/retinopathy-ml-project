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
6. Tune the four strongest models:
   ```powershell
   python tune_models.py --iterations 40 --folds 5
   ```

The tuning command keeps 20% of the data as an untouched final test set.
Hyperparameters are selected by stratified cross-validation on the remaining
training data. Decision thresholds are selected from out-of-fold training
probabilities (with recall >= 0.85 by default), never from the test set.

For a quick smoke test, use:
```powershell
python tune_models.py --iterations 2 --folds 3 --jobs 1
```

> If you run `python load_data.py` without the venv activated, Windows may use your system Python and fail to import `scipy`.

## Requirements

- Python 3.x
- `scipy`
- `pandas`
- TensorFlow/PyTorch
- NumPy, scikit-learn

## License

MIT License
