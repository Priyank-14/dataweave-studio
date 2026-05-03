![Python](https://img.shields.io/badge/Python-3.x-blue)
![GUI](https://img.shields.io/badge/GUI-Tkinter-green)
![Data Science](https://img.shields.io/badge/Data-Preprocessing-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🧵 DataWeave Studio

Transform raw data into clean, structured insights through an interactive GUI pipeline.

DataWeave Studio is a modern **GUI-based data preprocessing application** built with Python to simplify data cleaning, transformation, analysis, and visualization — all without writing complex code.

---

## 🚀 Overview

DataWeave Studio provides an end-to-end pipeline for working with datasets:

* Clean messy real-world data
* Transform and normalize values
* Reduce dimensionality (PCA & LDA)
* Visualize patterns and statistics
* Export processed datasets

All operations are performed through a **simple and interactive desktop interface**.

---

## 🖥️ UI Preview

### 🔹 Main Dashboard

![Dashboard](screenshots/dashboard.png)

### 🔹 Data Cleaning

![Cleaning](screenshots/cleaning.png)

### 🔹 Normalization

![Normalization](screenshots/normalization.png)

### 🔹 Visualization

![Visualization](screenshots/visualization.png)

### 🔹 Data Reduction (PCA / LDA)

![Reduction](screenshots/reduction.png)

---

## ✨ Features

### 🧹 Data Cleaning Pipeline

* Detects and replaces null-like values (`N/A`, `null`, `-`, etc.)
* Removes empty rows and duplicate entries
* Converts numeric-like strings into proper numeric format
* Handles missing values:

  * Median (numerical columns)
  * Mode / `"Unknown"` (categorical columns)
* Removes outliers using IQR (Interquartile Range)

---

### 📐 Data Transformation

* Min-Max Normalization
* Scales numeric data to range **[0, 1]**
* Handles constant columns safely

---

### 📉 Data Reduction

* **PCA (Principal Component Analysis)**
* **LDA (Linear Discriminant Analysis)**

---

### 📊 Data Visualization

* Statistical summary:

  * Mean, Median, Mode, Standard Deviation
* Graphs:

  * Bar Charts
  * Box Plots
  * Histogram
* Embedded visualization directly inside GUI

---

### 📂 File Handling

* Upload CSV datasets
* Preview:

  * Original data
  * Cleaned data
  * Normalized data
* Export cleaned dataset

---

## 🔄 Workflow

Upload Data → Clean Data → Normalize → Reduce → Visualize → Export

---

## 🛠️ Tech Stack

| Category         | Tools         |
| ---------------- | ------------- |
| Language         | Python        |
| GUI              | Tkinter       |
| Data Processing  | Pandas, NumPy |
| Machine Learning | Scikit-learn  |
| Visualization    | Matplotlib    |

---

## 📦 Installation

```bash
git clone https://github.com/Priyank-14/dataweave-studio.git
cd dataweave-studio
pip install -r requirements.txt
python app.py
```

---

## 📊 Dataset

* Source: data.gov.in
* Format: CSV

---

## 👨‍💻 Author

Priyank Sinha
B.Tech CSE | AI/ML Enthusiast

---

## ⭐ Support

If you like this project, consider giving it a star ⭐
