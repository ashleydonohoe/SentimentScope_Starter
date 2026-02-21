
# 🎬 Udacity SentimentScope: Transformer-Based Sentiment Analysis on IMDB Reviews

A machine learning project that trains and evaluates a **transformer-based model** to classify movie reviews as **positive or negative** using the IMDB dataset.

This project simulates a real-world ML engineering task at a fictional streaming company (“Cinescope”) to improve recommendation systems through sentiment classification.

---

## 📌 Project Overview

This notebook walks through the full machine learning pipeline:

* Loading and preprocessing raw text review data
* Building a dataset and dataloaders
* Training a transformer model from scratch
* Evaluating performance on test data
* Generating predictions for unseen reviews

The goal is to develop a working sentiment classifier that can be integrated into recommendation systems or review-analysis tools.

---

## 🧠 Skills Demonstrated

* NLP preprocessing
* Transformer architectures
* PyTorch datasets & dataloaders
* Model training loops
* Evaluation metrics
* Data visualization
* Debugging ML pipelines

---

## 🗂️ Repository Structure

```
SentimentScope/
│
├── SentimentScope_starter.ipynb   # Main notebook (training + evaluation)
├── README.md                      # Project documentation
└── data/                          # IMDB dataset (downloaded or extracted)
```

---

## 📊 Dataset

**IMDB Movie Reviews Dataset**

* 50,000 labeled reviews
* Binary sentiment: positive or negative
* Pre-split into:

  * Training set
  * Testing set

The dataset is provided in compressed form (`aclIMDB_v1.tar.gz`) and extracted within the notebook.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sentimentscope.git
cd sentimentscope
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn tqdm
```

*(Add transformers if used in your final model)*

```bash
pip install transformers
```

---

## 🚀 How to Run

1. Open the notebook:

```bash
jupyter notebook
```

2. Run:

```
SentimentScope_starter.ipynb
```

3. Follow cells in order:

   * Load dataset
   * Preprocess text
   * Create dataset class
   * Train model
   * Evaluate performance

---

## 🏗️ Model Pipeline

### 1. Data Loading

* Reads raw `.txt` review files
* Converts into Pandas DataFrames
* Labels:

  * 1 → Positive
  * 0 → Negative

### 2. Preprocessing

* Tokenization
* Padding
* Vocabulary mapping
* Tensor conversion

### 3. Model

Transformer-based classifier using:

* Embeddings
* Attention layers
* Fully connected output

### 4. Training

* Cross-entropy loss
* Optimizer (Adam)
* Batch training
* Validation loop

### 5. Evaluation

* Accuracy
* Loss curves
* Test predictions

---

## 📈 Example Use Cases

* Review sentiment detection
* Recommendation engines
* Social media monitoring
* Product feedback analysis
* Finance/news sentiment tools

---

## 🧑‍💻 Author

**Ashley Donohoe**: Freelance writer, editor, and data/AI learner

Focus: finance, AI, and machine learning applications

---

## 📜 License

This project is for educational and portfolio use.
