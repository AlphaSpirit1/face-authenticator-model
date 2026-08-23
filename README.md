# AI-Based Authentic Face Image Classification and Verification

This project implements a face-authenticity detection pipeline for the major project described in the attached brief. It combines image-based features with metadata such as gender, age group, confidence, and image quality to classify faces as authentic or manipulated.

## Project goals

- Preprocess the facial image dataset and metadata.
- Train a binary classifier that distinguishes REAL vs FAKE images.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Analyze metadata impact on classification performance.
- Support future deepfake detection and image-verification workflows.

## Project structure

- `src/face_authenticity/data.py` – dataset loading, metadata preparation, and split generation.
- `src/face_authenticity/model.py` – PyTorch CNN classifier and evaluation utilities.
- `src/face_authenticity/train.py` – training pipeline and model persistence.
- `examples/create_demo_dataset.py` – synthetic demo dataset generator for quick validation.
- `artifacts/` – saved models and evaluation outputs.

## Expected dataset layout

Place the dataset in a folder with:

- `metadata.csv` containing columns such as `image_id`, `label`, `gender`, `age_group`, `quality_score`, `confidence`, `difficulty`, and `dataset_split`.
- A folder of images. If the CSV includes a file path column, the code uses it. Otherwise it expects files named like `image_id.jpg` in the image directory.

## Quick start

### Full dataset run (recommended for the final project)

Use these commands when working with the full provided CSV and the corresponding image directory.

1. Open PowerShell in the project folder:
   ```powershell
   cd C:\Users\harsh\Documents\model
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Set the Python path so the package resolves correctly:
   ```powershell
   $env:PYTHONPATH = ".\src"
   ```

4. Filter out the invalid tail rows and keep only valid local-image rows before training:
   ```powershell
   @'
   import pandas as pd
   from pathlib import Path

   csv_path = Path("AI_Classification-Project.csv")
   df = pd.read_csv(csv_path)

   # Remove rows after the valid image range
   df = df[df["image_id"] <= 5557].copy()

   image_dir = Path("dataset/images")
   if image_dir.exists():
       valid_ids = {f.stem for f in image_dir.iterdir() if f.is_file()}
       df = df[df["image_id"].astype(str).isin({str(v) for v in valid_ids})].copy()

   df.to_csv("AI_Classification-Project_filtered.csv", index=False)
   print("rows after filtering:", len(df))
   print("split counts:", df["dataset_split"].value_counts().to_dict())
   '@ | python -
   ```

5. Train the model on the filtered full dataset:
   ```powershell
   python -m face_authenticity.train `
     --metadata-csv .\AI_Classification-Project_filtered.csv `
     --images-dir .\dataset\images `
     --output-dir .\artifacts\final_submission `
     --epochs 10 `
     --batch-size 32 `
     --learning-rate 3e-4 `
     --split-column dataset_split
   ```

6. Test the trained model on the filtered dataset and print evaluation metrics:
   ```powershell
   python -m face_authenticity.infer `
     --model-path .\artifacts\final_submission\face_auth_model.pt `
     --metadata-csv .\AI_Classification-Project_filtered.csv `
     --images-dir .\dataset\images `
     --batch-size 16
   ```

7. Optional: save the test metrics to a JSON file:
   ```powershell
   python -m face_authenticity.infer `
     --model-path .\artifacts\final_submission\face_auth_model.pt `
     --metadata-csv .\AI_Classification-Project_filtered.csv `
     --images-dir .\dataset\images `
     --batch-size 16 `
     --output-json .\artifacts\final_submission\test_metrics.json
   ```

### Demo smoke test

Use this only for checking that the training pipeline works on synthetic data.

```powershell
python examples/create_demo_dataset.py --output-dir examples/demo_dataset --n-samples 200
python -m face_authenticity.train --metadata-csv examples/demo_dataset/metadata.csv --images-dir examples/demo_dataset/images --output-dir artifacts/demo_run --epochs 3 --batch-size 16
```

## Model details

The classifier is a compact CNN trained on resized RGB facial images, with a metadata branch that incorporates demographic and quality attributes. The final layer produces a probability estimate for authenticity.

## Evaluation metrics

The training script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Loss per epoch
- Confusion matrix

## Notes

- Image filenames must match metadata identifiers when using the default lookup logic.
- If a dataset split column is not present, the code automatically creates train/validation/test splits.
- The project is ready to extend with deeper CNN architectures or transfer learning if a larger dataset is available.
