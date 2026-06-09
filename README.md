# Toxic-Comment-Classification
# Multi-Label Toxic Comment Detection Framework
> A Multi-Architectural Optimization Framework Using Semantic and Bidirectional Attention Networks.

---

## 👥 Project Overview & Authors
* **members:** Iqra Hamayoun, Eman Aslam
* **Program:** Bachelor of Science in Software Engineering (BSSE)
* **Institution:** University of Management and Technology (UMT), Sialkot Campus
* **Course:** Natural Language Processing (NLP)

---

## 📌 Project Objectives
Online communication channels require automated moderation pipelines to preserve safe digital spaces. Abusive interactions typically manifest across multiple overlapping toxicity types (e.g., a comment can simultaneously be an *insult* and *obscene*). 

This framework maps out the development, benchmarking, and engineering trade-offs of three generations of NLP architectures to resolve multi-label toxic comment classification across **6 target attributes**:
1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

---

## 🏗️ Comparative Architectures Evaluated
Our framework implemented and evaluated three evolutionary phases of NLP models to handle severe data imbalance:

1. **Model I: Classical Statistical Baseline (TF-IDF)**
   * Extracts global lexical counts via Term Frequency-Inverse Document Frequency.
2. **Model II: Deep Learning Sequential Network (Bi-LSTM)**
   * Uses a custom Embedding layer, `Bidirectional(LSTM)` recurrent units, and dropout regularization.
3. **Model III: Contextual Transformer Attention (Small-BERT)**
   * Fine-tuned transformer utilizing bidirectional self-attention configurations through TensorFlow Hub (`bert_en_uncased_L-4_H-512_A-8`).

---

## 📈 Performance Benchmarking Summary

| Model Architecture | Validation Accuracy | Structural Performance (Macro F1) | Inference Latency |
| :--- | :--- | :--- | :--- |
| **Model I: TF-IDF Baseline** | Moderate | 0.40 | Near-Instantaneous |
| **Model II: Sequential Bi-LSTM** | Moderate-High | ~0.36 | Low-Moderate |
| **Model III: Small-BERT** | **96.97%** | **0.58** | Balanced (~79-81ms) |

### Key Takeaways
* **The Accuracy vs. F1 Trade-off:** While simple models score high on baseline validation accuracy by guessing the majority "clean text" class correctly, they fail to classify minority toxic labels (like threats). 
* **The Transformer Advantage:** Small-BERT exhibits a massive leap in **Macro F1-Score (0.58)**, proving that self-attention layers are far better at capturing subtle toxic cues hidden in highly imbalanced datasets.

---

## 💾 Model Serialization & Verification
To ensure production readiness, the fine-tuned BERT Transformer weights are fully serialized to disk:
```python
model_3.save("toxic_comment_bert_model.keras")
