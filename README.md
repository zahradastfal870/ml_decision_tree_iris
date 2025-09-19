# Machine Learning Project: Decision Tree on Iris Dataset

This project demonstrates training a Decision Tree classifier on the Iris dataset 
using scikit-learn with different maximum depths (1, 2, 3).

---

## Student Info
- Name: Zahra Dastfal  
- Student ID: 700777425  

---

## How to Run
Install dependencies and run the script:
```bash
pip install scikit-learn
python decision_tree.py

## Results

| Depth | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| 1     | 0.6667         | 0.6667        |
| 2     | 0.9714         | 0.8889        |
| 3     | 0.9810         | 0.9778        |


## Discussion (Underfitting vs Overfitting)
- Depth = 1 → Underfitting (too simple, low accuracy).  
- Depth = 2 → Good balance, still generalizes well.  
- Depth = 3 → Best balance, high train and test accuracy.  

