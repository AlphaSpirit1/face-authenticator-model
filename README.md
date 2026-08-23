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

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a demo dataset and run a smoke test:
   ```bash
   python examples/create_demo_dataset.py --output-dir examples/demo_dataset --n-samples 200
   python -m src.face_authenticity.train --metadata-csv examples/demo_dataset/metadata.csv --images-dir examples/demo_dataset/images --epochs 3 --batch-size 16 --output-dir artifacts/demo_run
   ```

3. Use a real dataset:
   ```bash
   python -m src.face_authenticity.train --metadata-csv path/to/metadata.csv --images-dir path/to/images --epochs 20 --batch-size 32 --output-dir artifacts/final_run
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
