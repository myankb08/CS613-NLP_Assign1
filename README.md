# CS613-NLP_Assign1: IndicTrans2 Reproduction Study

🔬 **A Comprehensive Reproduction and Analysis of the IndicTrans2 Multilingual Translation Model**

## 📋 Project Overview

This repository contains a detailed reproduction study of the **IndicTrans2** model, specifically evaluating the distilled M2M variant (`indictrans2-indic-indic-dist-320M`) on the IN22-Gen benchmark. The study investigates the reproducibility of state-of-the-art multilingual machine translation models and identifies potential factors affecting performance consistency.

### Technical Specifications
- **Model**: `ai4bharat/indictrans2-indic-indic-dist-320M`
- **Dataset**: IN22-Gen test split (1024 parallel sentences)
- **Languages Evaluated**: Hindi, Tamil, Bengali, Marathi, Telugu
- **Evaluation Metric**: chrF++ (sacrebleu implementation)
- **Environment**: Kaggle (Tesla T4 GPU, transformers==4.53.2)

## 📊 Experimental Results

### Performance Comparison Summary

| Metric | Paper Scores | Our Reproduction | Difference |
|--------|--------------|------------------|------------|
| Average xx→lang | 43.5 | 34.5 | **-9.0** |
| Average lang→xx | 41.4 | 34.5 | **-6.9** |

### Key Findings

**Consistent Performance Gap**: All five evaluated languages showed **7-10 point lower chrF++ scores** across both translation directions compared to the original paper's reported results. This consistent pattern suggests systematic differences rather than random variation.

### Detailed Language-wise Results

![Detailed Results Table](https://github.com/user-attachments/assets/fb10b38e-03b4-4226-9182-650cca237c66)

*Table: Detailed performance comparison across all evaluated languages*

## 🔍 Methodology Insights

### Two-Phase Evaluation Approach

Our investigation employed a comprehensive two-phase evaluation strategy that revealed crucial methodological insights:

1. **Initial Evaluation (200 samples)**:
   - Showed 13-16 point performance gap
   - Later identified as misleading due to sampling bias

2. **Full Dataset Evaluation (1024 samples)**:
   - Revealed more accurate 7-10 point gap
   - Demonstrated the importance of comprehensive evaluation

![200 Sample Results](https://github.com/user-attachments/assets/0c78797f-88e6-4c74-bc50-5925de43eb3c)

*Comparison: 200-sample vs. full dataset evaluation results*

### Reference: Original Paper Results
![Original Paper Table](https://github.com/user-attachments/assets/20b8696a-fc04-45e0-b2d2-054f1e2a0db1)

*Source: Original IndicTrans2 paper results (Table 20)*

## 🎯 Critical Discoveries

### 1. **Sampling Bias in Evaluation**
Our two-phase evaluation revealed that **small sample sizes significantly underestimate model performance**. The 200-sample evaluation showed a 13-16 point gap, while the full 1024-sample evaluation revealed a more accurate 7-10 point gap. This highlights the critical importance of using complete datasets for reliable model comparisons.

### 2. **Consistent Performance Patterns**
The performance gap was observed consistently across all language directions and translation tasks, indicating systematic rather than random factors affecting the results.

### 3. **Scope Limitations**
Due to computational constraints, this study focused on 5 of the 16 languages mentioned in the original paper. Future work could explore whether evaluating all languages might affect the overall chrF++ scores.

## 🔬 Potential Factors for Performance Gap

Based on our analysis, several factors may contribute to the observed performance differences:

### Technical Factors
1. **Library Version Differences**
   - Our study used transformers==4.53.2 for compatibility
   - The original paper may have used different library versions
   - Version differences can affect model behavior and tokenization

2. **Preprocessing Variations**
   - Potential slight differences in text normalization pipelines
   - Variations in sentence segmentation and tokenization
   - Differences in handling special characters and diacritics

3. **Tokenizer Behavior**
   - Potential differences in tokenizer configuration or behavior
   - Variations in vocabulary handling for low-resource languages

### Methodological Factors
4. **Evaluation Protocol**
   - Potential differences in inference parameters (beam size, length penalty)
   - Variations in batch processing and memory optimization
   - Hardware and precision differences (FP16 vs. FP32)

5. **Dataset Considerations**
   - Possible test set contamination in original evaluation
   - Differences in data cleaning and preprocessing pipelines

## 🚀 Implementation Details

### Technical Setup
```python
# Core components used in reproduction
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-indic-indic-dist-320M",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
ip = IndicProcessor(inference=True)  # From IndicTransToolkit
