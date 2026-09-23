# Skin Lesion Classification Model

## Project Overview

This project was developed as part of the Machine Learning Student Network (MLSN).

## Goal
The goal of the project was to develop a machine learning model capable of classifying dermoscopic images of skin lesions.

The project combines:

* Computer vision
* Deep learning
* Image classification
* Patient metadata
* Exploratory data analysis
* Streamlit deployment

The trained model classifies images into seven lesion categories from the HAM10000 dataset.

## Dataset

The project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset from Kaggle.

The dataset contains **10,015 dermatoscopic images** of pigmented skin lesions across seven diagnostic categories.

### Classes

| Label   | Description                                   |
| ------- | --------------------------------------------- |
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `bcc`   | Basal cell carcinoma                          |
| `bkl`   | Benign keratosis-like lesions                 |
| `df`    | Dermatofibroma                                |
| `mel`   | Melanoma                                      |
| `nv`    | Melanocytic nevi                              |
| `vasc`  | Vascular lesions                              |

The dataset also contains metadata including:

* Lesion ID
* Image ID
* Diagnosis
* Diagnosis type
* Age
* Sex
* Localization

The dataset contains multiple diagnostic verification methods, including histopathology.

## Exploratory Data Analysis

Exploratory analysis was performed to understand the dataset and identify potential class imbalance.

The analysis included:

* Class distribution
* Age distribution
* Sex distribution
* Metadata correlations
* Example lesion images
* Training-set class balance

A major observation was the strong imbalance between lesion classes. **Melanocytic nevi (`nv`) represents the dominant class**, while several other classes have substantially fewer examples.

This imbalance affects model performance because overall accuracy can be influenced by the majority class.

The project also examined the relationship between age and the missing-age indicator as part of the metadata analysis.

## Data Preprocessing

### Metadata Processing

Missing age values were handled by:

1. Creating an `age_missing` indicator feature.
2. Replacing missing age values with the median age.

```python
df['age_missing'] = df['age'].isna().astype(int)

median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)
```

The model therefore receives two metadata features:

* Patient age
* Whether the original age value was missing

### Dataset Split

The dataset was divided using stratified sampling:

| Split      | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Test       |        15% |

Stratification was used to preserve the class distribution across the three splits.

### Image Preprocessing

Images were processed using the following transformations:

* Resize to 256 pixels
* Center crop to 224 × 224
* Convert to tensors
* Normalize using ImageNet mean and standard deviation

Training images also received random horizontal flipping to introduce additional variation.

```python
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Model Architecture

The project uses a **dual-branch CNN architecture** built from scratch using PyTorch.

Rather than relying only on the image, the model combines visual features with patient age information.

### Architecture

```text
                    Dermoscopic Image
                           │
                           ▼
                    Convolutional CNN
                           │
                           ▼
                    Image Features
                           │
                           │
                           ├──────────────┐
                           │              │
                           │         Patient Age
                           │              │
                           │              ▼
                           │        Metadata Branch
                           │              │
                           │              ▼
                           │       Metadata Features
                           │              │
                           └───────┬──────┘
                                   ▼
                              Concatenate
                                   │
                                   ▼
                              Classifier
                                   │
                                   ▼
                         7 Lesion Categories
```

### Image Branch

The image branch contains three convolutional blocks:

* Conv2D: 3 → 32 channels
* ReLU
* Max Pooling
* Conv2D: 32 → 64 channels
* ReLU
* Max Pooling
* Conv2D: 64 → 128 channels
* ReLU
* Max Pooling

The extracted image features are then passed through:

* Flatten
* Fully connected layer: 128 × 28 × 28 → 512
* ReLU
* Dropout (0.5)

### Metadata Branch

The metadata branch takes two features:

* Age
* Age-missing indicator

These are passed through:

```text
2 inputs
   ↓
Linear(2 → 32)
   ↓
ReLU
```

### Final Classifier

The 512 image features and 32 metadata features are concatenated:

```text
512 image features + 32 metadata features
                    ↓
              544 features
                    ↓
          Linear(544 → 7)
                    ↓
          Seven lesion classes
```

The architecture is implemented in PyTorch.

## Training

The model was trained using:

* **Framework:** PyTorch
* **Loss:** Cross Entropy Loss
* **Optimizer:** Adam
* **Learning Rate:** 0.0001
* **Batch Size:** 32
* **Epochs:** 10

Training and validation performance were monitored after each epoch.

## Results

The model achieved approximately:

* **Training Accuracy:** 79%
* **Validation Accuracy:** 77%

Training loss decreased over the training process, while validation loss also generally decreased.

The project also evaluated the model on a held-out test set and generated:

* Confusion matrix
* Classification report
* Per-class accuracy

These metrics were used to examine performance beyond overall accuracy and identify differences between lesion classes.

## Class Imbalance

One of the primary challenges in the project was the imbalance in the HAM10000 dataset.

The `nv` class contains substantially more examples than several other lesion categories. As a result, the model showed stronger prediction performance for the dominant class while some smaller classes had lower recall.

The confusion matrix also showed that melanoma (`mel`) could be confused with melanocytic nevi (`nv`).

This highlights why overall accuracy alone is not sufficient for evaluating an imbalanced medical image classification problem.

## Streamlit Application

The trained model was integrated into a Streamlit application to provide an interactive interface.

The application allows a user to:

1. Upload a dermoscopic image.
2. Enter a patient age.
3. Run the trained model.
4. View the model's top three predicted lesion categories.
5. View the corresponding prediction confidence scores.

### Application Pipeline

```text
Upload Image
      ↓
Resize & Center Crop
      ↓
Normalize Image
      ↓
Enter Patient Age
      ↓
CNN Image Branch + Metadata Branch
      ↓
Combine Features
      ↓
Seven-Class Prediction
      ↓
Display Top 3 Results
```

The Streamlit application loads the saved PyTorch model weights from `skin_cancer_model.pth`.

The app applies the same evaluation preprocessing used during model development before generating predictions.

## Project Structure

```text
.
├── skin_cancer_classifier.ipynb
├── app.py
├── skin_cancer_model.pth
├── requirements.txt
└── README.md
```

### Files

**`skin_cancer_classifier.ipynb`**
Google Colab notebook containing dataset preparation, exploratory data analysis, preprocessing, model development, training, and evaluation.

**`app.py`**
Streamlit application for uploading dermoscopic images and generating predictions.

**`skin_cancer_model.pth`**
Saved PyTorch model weights.

**`requirements.txt`**
Python dependencies required to run the project.

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Pillow
* Scikit-learn
* Streamlit
* Google Colab
* Kaggle
* GitHub

## Running the Application

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

### 4. Make a Prediction

1. Upload a JPG or PNG dermoscopic image.
2. Enter the patient's age.
3. Select **Run Diagnostic Analysis**.
4. Review the model's top three predicted categories and confidence scores.

## Future Improvements

### Address Class Imbalance

* Apply class weighting during training.
* Experiment with oversampling minority classes.
* Explore targeted data augmentation.
* Evaluate additional imbalance-handling techniques.

### Improve the Model

* Experiment with deeper CNN architectures.
* Compare the custom CNN against transfer-learning approaches.
* Fine-tune pretrained computer vision models.
* Tune learning rate, batch size, and network architecture.
* Use additional evaluation metrics such as macro F1-score and balanced accuracy.

### Expand Dataset Diversity

Future versions could incorporate datasets with greater diversity in skin tones and image acquisition conditions.

This would allow the model to be evaluated across a broader range of real-world imaging conditions.

### Application Improvements

* Add clearer prediction explanations.
* Display all seven class probabilities.
* Improve the user interface.
* Add confidence thresholds.
* Improve model inference speed.
* Provide more detailed information about each predicted class.
ulting Streamlit application provides an accessible interface for testing the trained model and visualizing its predictions while highlighting the challenges of class imbalance and generalization in medical image classification.
