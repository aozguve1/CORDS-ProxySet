# CS_436_Final_Project_Cords
Official repository for the CS436 Final Project: 'Proxy-Set Selection'. Implementing a lightweight, loss-based data selection strategy on CIFAR-10 using ResNet-18 to minimize computational overhead in adaptive learning.

# Efficient & Robust Data Selection for Sustainable Deep Learning: A Proxy-Set Approach

This repository contains the implementation of **Proxy-Set Selection**, a novel data selection strategy designed to make Deep Learning training more energy-efficient and faster **without compromising accuracy**.  
This project replicates state-of-the-art methods (**GradMatch**, **GLISTER**) and introduces a robust improvement to overcome their computational bottlenecks.

---

## 🎯 Project Goal: Green AI

Training modern deep learning models is computationally expensive and energy-intensive.  
Existing adaptive data selection methods (e.g., **GradMatch**) aim to reduce training time by selecting informative subsets, but they often introduce **massive computational overhead** during the selection phase (full dataset scans).

### **Our Solution: Proxy-Set Selection**
A hybrid strategy that combines:
- the **speed** of Random Sampling  
- the **intelligence** of Loss-Based Selection

---

## 📊 Key Results

Performance measured on **CIFAR-10 with ResNet-18** using an **Apple M4** chip.

| Metric               | Standard GradMatch | Proxy-Set (Ours) | Improvement |
|---------------------|--------------------|------------------|-------------|
| Avg. Selection Time | ~70 sec            | ~11 sec          | **6× Faster 🚀** |
| Carbon Emissions    | 0.0107 kg          | 0.0088 kg        | **17% Less 🌱** |
| Test Accuracy       | 81.5%              | 80.7%            | **<1% Drop ✅** |

---

## 🛠️ Implementation & Modifications

This project builds upon the **CORDS** repository.  
Below are the specific changes and contributions.

---

### New Strategy: Proxy-Set Selection

**File Created:**  
`cords/utils/data/dataloader/SL/adaptive/proxy_gradmatchdataloader.py`

**Key Logic:**

- **Proxy Pool:** Instead of scanning full dataset (N), sample a proxy subset (30% of N).
- **Temporary Loader:** Use a disposable DataLoader to avoid PyTorch dataset immutability errors.
- **Robust Selection:** Perform forward pass on the proxy pool to compute losses and pick the “hardest” examples.

**Outcome:**  
Bypasses silent failures + eliminates selection latency in GradMatch.

---

### CodeCarbon Integration

**File Modified:**  
`examples/SL/image_classification/python_notebooks/CORDS_SL_CIFAR10_Custom_Train `

**Changes:**
- Added `EmissionsTracker` before training loop  
- Injected metadata (accuracy) into `emissions.csv`  
- Enables full energy-tracking workflow

---

**Hardware Adaptation:**  
- Updated device logic to support **Apple Silicon (MPS)** in addition to CUDA/CPU

---
### **Abdullah Salih Özgüven**
- **Experimental Execution:**  
  Leveraged high-performance local hardware (Apple M4) to run all computational experiments, including replication of baselines (**GLISTER**, **GradMatch**) and evaluation of the **Proxy-Set** strategy.
- **Strategy Design & Optimization:**  
  Analyzed relevant literature to determine optimal hyperparameters and configured adaptive strategies for maximum efficiency.
- **Implementation:**  
  Developed the custom `ProxyGradMatchDataLoader`, integrated **CodeCarbon**, and resolved multiple CORDS library bugs.

---

### **Ranjith Crystal Daniel**
- **Literature Review:**  
  Conducted an extensive review of data subset selection methods (GradMatch, CRAIG, GLISTER) to establish a strong theoretical foundation.
- **Problem Analysis:**  
  Investigated the root theoretical causes of instability issues in the standard CORDS implementation, helping guide the need for a robust alternative.


# 📚 References

- **GradMatch:**  
  Killamsetty et al., *Grad-Match: Gradient Matching based Data Subset Selection for Efficient Deep Model Training*, ICML 2021.

- **GLISTER:**  
  Killamsetty et al., *GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning*, AAAI 2021.

- **CodeCarbon:**  
  Schmidt et al., *CodeCarbon: Estimate and Track Carbon Emissions from Machine Learning Computing*, 2021.

## How to Run


### 1. Installation

```bash
cd CS436_Final_Project_Cords
pip install -r requirements.txt
pip install codecarbon dotmap apricot-select
```
# 📚 References

- **GradMatch:**  
  Killamsetty et al., *Grad-Match: Gradient Matching based Data Subset Selection for Efficient Deep Model Training*, ICML 2021.

- **GLISTER:**  
  Killamsetty et al., *GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning*, AAAI 2021.

- **CodeCarbon:**  
  Schmidt et al., *CodeCarbon: Estimate and Track Carbon Emissions from Machine Learning Computing*, 2021.


