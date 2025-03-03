# Knowledge Tracing: LSTM vs. Transformer Models  

## 📌 Overview  
This project explores **Deep Knowledge Tracing (DKT)** using **LSTM-based and Transformer-based models** on the **ASSISTments 2017 dataset**. The study evaluates the impact of **interaction encoding strategies, sequence lengths, and sliding window strides** on model performance.  

## 📊 Key Findings  
- **LSTMs outperform Transformers on shorter sequences** due to better temporal modeling.  
- **Transformers require longer sequences** to leverage self-attention effectively.  
- **Dense overlapping sequences (stride = 1) improve learning** by increasing effective training data.  
- **Combined encoding (single integer) slightly outperforms separate encoding**, reducing redundancy.  

## 🚀 Features  
✔ LSTM-based DKT model implementation  
✔ Transformer-based Knowledge Tracing model  
✔ Sequence generation using **sliding window approach**  
✔ **Early stopping** and **AUC-based evaluation**  
✔ **Visualization**: Prediction heatmaps and AUROC curves  

## 📂 Dataset  
We use the **ASSISTments 2017 dataset**, a widely used benchmark for **Knowledge Tracing research**. It includes:  
- **Student response records**  
- **Skill ID mappings**  
- **Timestamps and correctness labels**  
