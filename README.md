# Classification Algorithms Showcase: From Perceptron to Random Forests

A comprehensive exploration of fundamental machine learning classification algorithms using scikit-learn, with detailed visualizations and comparisons on the Iris dataset.

## 📁 Project Structure

```
├── classification_algorithms_comparison.py  # Main Python script with all exercises
├── decision_boundary_visualizer.py          # Utility functions for plotting
├── README.md                                # This file
└── requirements.txt                         # Python dependencies
```

## 🎯 Overview

This project implements and compares 7 key classification algorithms on the classic Iris dataset, providing educational visualizations and insights into each algorithm's behavior, strengths, and limitations.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/classification-showcase.git
cd classification-showcase
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy matplotlib scikit-learn pandas
```

## 📊 Algorithms Implemented

### **1. Perceptron** (`Exercise 2`)
- Linear binary classifier
- Exploration of learning rates (η = 0.01, 0.1, 1.0)
- Demonstration of limitations on non-linear data (moons dataset)

### **2. Logistic Regression** (`Exercise 3`)
- Multi-class probabilistic classifier
- Regularization strength analysis (C = 0.01, 1.0, 100.0)
- Probability predictions and decision boundaries

### **3. Support Vector Machines** (`Exercises 4-5`)
- Linear SVM with varying C values
- RBF kernel SVM with gamma tuning (γ = 0.01, 1.0, 100.0)
- Support vector identification and visualization

### **4. Decision Trees** (`Exercise 6`)
- Gini impurity criterion
- Tree depth control (max_depth=4)
- Complete tree visualization with scikit-learn's `plot_tree`

### **5. Random Forests** (`Exercise 7`)
- Ensemble of decision trees (25, 100 trees)
- Out-of-bag (OOB) error estimation
- Feature importance analysis

### **6. K-Nearest Neighbors** (`Exercise 8`)
- Distance metrics (Manhattan: p=1, Euclidean: p=2)
- k-value analysis (k=1, 5, 10)
- Discussion of underfitting vs. overfitting

### **7. Hyperparameter Tuning** (`Exercise 9`)
- GridSearchCV for systematic parameter optimization
- Cross-validation performance comparison
- Final model evaluation on test set

## 📈 Key Visualizations

1. **Decision Boundary Plots**: Custom `plot_decision_regions()` function
2. **Learning Curves**: Perceptron error convergence
3. **Tree Visualizations**: Complete decision tree structure
4. **Comparative Analysis**: Side-by-side algorithm performance

## 🎓 Educational Insights

### **Algorithm Strengths & Weaknesses:**
- **Perceptron**: Fast but only works for linearly separable data
- **Logistic Regression**: Provides probabilities but assumes linear decision boundaries
- **SVM**: Effective for high-dimensional data but sensitive to parameters
- **Decision Trees**: Interpretable but prone to overfitting
- **Random Forests**: Reduces variance but less interpretable
- **KNN**: Simple but computationally expensive for large datasets

### **Key Findings:**
- **Perceptron**: 100% accuracy on Iris binary classification, but fails on moons dataset (68% accuracy)
- **Logistic Regression**: Best regularization found at C=0.1 with 95.2% CV score
- **RBF SVM**: Overfitting occurs at high gamma values (γ=100.0)
- **Random Forests**: Feature importance shows petal dimensions are most informative
- **KNN**: Optimal performance at k=1 with Manhattan distance

## 📋 Requirements

- Python 3.7+
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- pandas >= 1.2.0

## 📝 Code Example

```python
# Using the decision boundary visualization utility
from decision_boundary_visualizer import plot_decision_regions

# Train any classifier
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_std, y_train)

# Visualize decision regions
plot_decision_regions(X_combined_std, y_combined, classifier=svm,
                      test_idx=range(len(X_train), len(X_combined_std)))
plt.title('SVM Decision Boundary')
plt.show()
```

## 📊 Dataset

### **Iris Dataset (Primary)**
- 150 samples, 4 features (using petal length & width)
- 3 classes: Setosa, Versicolor, Virginica
- 70/30 train-test split with stratification

### **Moons Dataset (Non-linear Example)**
- 200 samples with 25% noise
- Demonstrates Perceptron limitations
- 2 classes with non-linear separation

## 🚀 Usage

Run the complete analysis:
```bash
python classification_algorithms_comparison.py
```

Run individual exercises:
```python
# Import specific exercise
from classification_algorithms_comparison import exercise3_logistic_regression
exercise3_logistic_regression()
```

## 🔍 Detailed Analysis

### **Hyperparameter Effects:**
1. **C in SVM/Logistic Regression**: Controls regularization strength
2. **γ in RBF SVM**: Controls kernel width and model complexity
3. **k in KNN**: Balances bias-variance tradeoff
4. **max_depth in Decision Trees**: Controls overfitting

### **Performance Metrics:**
- All models achieved 97.8-100% accuracy on Iris test set
- Best cross-validation score: KNN (97.1%)
- Most interpretable: Decision Trees
- Best for non-linear data: RBF SVM

## 📚 Educational Value

This project serves as an excellent resource for:
- Understanding algorithm decision boundaries
- Visualizing hyperparameter effects
- Comparing linear vs. non-linear classifiers
- Learning ensemble methods
- Practicing model evaluation techniques

## 👥 Contributing

Contributions are welcome! Areas for improvement:
- Add more datasets (wine, digits, breast cancer)
- Implement additional algorithms (Naive Bayes, Neural Networks)
- Add ROC curves and precision-recall plots
- Create interactive visualizations with Plotly

## 📝 License

This project is open source and available under the MIT License.

---

**Note**: This project demonstrates that multiple algorithms can achieve similar performance on simple datasets, but their behavior, assumptions, and interpretability differ significantly - highlighting the importance of algorithm selection based on problem context rather than just accuracy scores.

---
