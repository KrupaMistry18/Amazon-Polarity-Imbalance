# Amazon Polarity Sentiment Classification

This project demonstrates various strategies for handling class imbalance in sentiment classification using the Amazon Polarity dataset. It compares the performance of several machine learning models under different sampling and weighting strategies.

## Features

- Loads and preprocesses the Amazon Polarity dataset
- Trains and evaluates four classifiers:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM
  - k-Nearest Neighbors
- Handles class imbalance using:
  - Random undersampling
  - Random oversampling
  - Class weighting
- Visualizes:
  - Accuracy comparison across models and strategies
  - Class distribution by sampling strategy
  - Confusion matrices for each model and strategy

## Requirements

- Python 3.7+
- pip

### Python Packages

Install dependencies with:

```bash
pip install pandas scikit-learn matplotlib seaborn imbalanced-learn datasets
```

## Usage

1. **Run the script:**

   ```bash
   python amazonpolarity.py
   ```

2. **What it does:**
   - Downloads and preprocesses the Amazon Polarity dataset.
   - Splits the data into training and test sets.
   - Trains and evaluates models on:
     - Original (balanced random sample)
     - Undersampled (majority class reduced)
     - Oversampled (minority class increased)
     - Class-weighted (model compensates for imbalance)
   - Prints classification reports for each model and strategy.
   - Plots:
     - Accuracy comparison
     - Class distribution
     - Confusion matrices

## File Structure

- `amazonpolarity.py` — Main script for data processing, training, evaluation, and visualization.

## Notes

- The script samples 100,000 reviews for efficiency.
- All visualizations are displayed using matplotlib and seaborn.
- The code is modular and can be extended to include more models or sampling strategies.

## References

- [Amazon Polarity Dataset on HuggingFace](https://huggingface.co/datasets/amazon_polarity)
- [scikit-learn documentation](https://scikit-learn.org/)
- [imbalanced-learn documentation](https://imbalanced-learn.org/) 