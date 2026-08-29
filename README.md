# Student Learning Style Classification

Comparison of **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)** algorithms for classifying student learning style preferences based on the **FSLSM (Felder-Silverman Learning Style Model)**, across 4 dimensions:
- Processing Information (Active vs Reflective)
- Perceiving Information (Sensing vs Intuitive)
- Receiving Information (Visual vs Verbal)
- Understanding Information (Sequential vs Global)

## Abstract
Understanding students' learning styles can help tailor teaching methods and learning media to be more effective. This research builds a learning style classification model based on the FSLSM model using questionnaire data from students, with features including demographic data (gender, parental income, home regency, high school major), academic data (GPA), and study habits (study duration, preferred learning media). The data was processed through cleansing and transformation stages, followed by class balancing using SMOTE on the training data, before three classification algorithms — Decision Tree, Random Forest, and SVM — were trained and tested using an 80:20 split scheme and validated using 5-fold cross-validation. Test results show that Random Forest consistently delivered the best performance among the three algorithms across all four FSLSM dimensions, with accuracy ranging from 64%–79%, followed by Decision Tree and SVM. Complete evaluation results (accuracy, precision, recall, F1-score) and confusion matrix visualizations are available in the `hasil/` folder.

## Folder Structure
```
klasifikasi-gaya-belajar-mahasiswa/
│
├── notebook/
│   ├── klasifikasi_gaya_belajar.ipynb   # Main notebook (preprocessing + modeling)
│   └── preprocessing.py                 # Module for data cleansing & transformation
│
├── dataset/
│   └── dataset_clean.csv                # Preprocessed dataset (respondent identity removed)
│
├── hasil/
│   ├── confusion_matrix/                # Confusion matrix per model
│   ├── decision_tree/                   # Decision tree visualizations
│   ├── grafik_lainnya/                  # Performance charts & model comparisons
│   └── evaluasi_model.csv               # Summary of accuracy, precision, recall, F1-score
│
├── requirements.txt
└── README.md
```

## How to Run
1. Clone this repository
   ```bash
   git clone https://github.com/zalana0/klasifikasi-gaya-belajar-mahasiswa.git
   cd klasifikasi-gaya-belajar-mahasiswa
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Open `notebook/klasifikasi_gaya_belajar.ipynb` in Jupyter/Google Colab, then run the cells in order.

## Brief Methodology
- **Data cleansing**: name case folding, duplicate data removal
- **Data transformation**: column renaming, FSLSM label mapping, major & regency standardization, label encoding, ordinal encoding for income & study duration, GPA encoding, multi-label binarization for learning media, normalization (StandardScaler)
- **Data split**: 80:20 stratified
- **Class balancing**: SMOTE (training data only)
- **Validation**: 5-fold cross-validation
- **Algorithms**: Random Forest (200 trees), Decision Tree (CART, max_depth=5), SVM (RBF, C=10)

## Privacy Note
The published dataset (`dataset_clean.csv`) **does not include** respondents' full names, emails, student ID numbers, place of birth, or date of birth, in order to protect the privacy of participating students.

## Author
Salzabila Triana ([@zalana0](https://github.com/salzabilatriana))

## License
This project was created for undergraduate thesis (skripsi) purposes.
