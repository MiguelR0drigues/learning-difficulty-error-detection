# Error Pattern Detection POC

A Proof of Concept demonstrating AI-based detection of error patterns in student text answers using NLP and semantic embeddings.

## Overview

This POC demonstrates that:
- ✅ Student errors follow identifiable patterns (not just "wrong")
- ✅ Patterns can be automatically detected using NLP techniques
- ✅ The approach is language-agnostic and extensible to other domains

## Key Results

- **Overall Accuracy**: 66.7% (28/42 correct predictions)
- **Perfect Detection**: Linguistic errors (100%) and vague answers (100%)
- **Model Used**: Sentence-BERT (`all-MiniLM-L6-v2`)

---

## How It Works: Detailed Explanation

### High-Level Architecture

```
┌─────────────────┐
│  Student Answer │
│   + Reference   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              PREPROCESSING MODULE                        │
│  • Lowercase conversion                                  │
│  • Whitespace normalization                              │
│  • Tokenization                                          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│           PARALLEL FEATURE EXTRACTION                    │
├──────────────────┬──────────────────┬───────────────────┤
│   Semantic       │    Keyword       │    Linguistic     │
│   Similarity     │    Analysis      │    Analysis       │
│                  │                  │                   │
│  Sentence-BERT   │  • Extract key   │  • Spell check    │
│  embeddings      │    concepts      │  • Count errors   │
│  • Reference     │  • Compare with  │  • Error ratio    │
│  • Student       │    student       │                   │
│  • Cosine sim.   │  • Coverage %    │                   │
└──────────┬───────┴────────┬─────────┴─────────┬─────────┘
           │                │                   │
           └────────────────┼───────────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │   HEURISTIC ERROR DETECTOR           │
         │   (Priority-based Decision Tree)     │
         └──────────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   ERROR CLASSIFICATION   │
              │   + Confidence Score     │
              │   + Explanation          │
              └─────────────────────────┘
```

### Detection Pipeline Flow

```
START
  │
  ├─► Load student answer & reference answer
  │
  ├─► [1] PREPROCESS TEXT
  │     │
  │     ├─ Normalize (lowercase, clean whitespace)
  │     └─ Tokenize (split into words)
  │
  ├─► [2] COMPUTE SEMANTIC SIMILARITY
  │     │
  │     ├─ Generate embedding for reference answer
  │     ├─ Generate embedding for student answer
  │     └─ Calculate cosine similarity (0-1 score)
  │
  ├─► [3] EXTRACT FEATURES
  │     │
  │     ├─ Word count
  │     ├─ Extract keywords from reference
  │     ├─ Check keyword coverage in student answer
  │     ├─ Count spelling errors
  │     └─ Calculate spelling error ratio
  │
  ├─► [4] APPLY DETECTION RULES (Priority Order)
  │     │
  │     ├─ CHECK: High similarity (≥0.75) + Good keyword coverage (≥50%)?
  │     │   YES → CLASSIFY AS: ✓ Correct (none)
  │     │   NO  → Continue ↓
  │     │
  │     ├─ CHECK: Word count < 5?
  │     │   YES → FLAG: Vague Answer (high priority)
  │     │
  │     ├─ CHECK: Spelling errors ≥ 3?
  │     │   YES → FLAG: Linguistic Error (high priority)
  │     │
  │     ├─ CHECK: Similarity < 0.55?
  │     │   YES → FLAG: Conceptual Error (wrong understanding)
  │     │
  │     ├─ CHECK: Keyword coverage < 40%?
  │     │   YES → FLAG: Missing Key Concepts
  │     │
  │     └─ No issues found?
  │         YES → CLASSIFY AS: ✓ Correct (none)
  │
  ├─► [5] SELECT PRIMARY ERROR
  │     │
  │     └─ Sort flagged issues by confidence score
  │         Return highest confidence prediction
  │
  ├─► [6] GENERATE EXPLANATION
  │     │
  │     ├─ Primary error explanation
  │     └─ Mention secondary issues if detected
  │
  └─► OUTPUT
        │
        ├─ Predicted error type
        ├─ Confidence score (0-1)
        └─ Human-readable explanation
```

---

## Error Types Detected

| Type | Description | Detection Method |
|------|-------------|-----------------|
| `none` | Correct answer | High similarity (≥0.75) + good keyword coverage (≥50%) |
| `conceptual_error` | Incorrect understanding of core concepts | Very low semantic similarity (<0.55) |
| `missing_key_concepts` | Omits essential terms/ideas | Low keyword coverage (<40%) with moderate similarity |
| `linguistic_error` | Spelling/grammar issues | ≥3 spelling errors detected |
| `vague_answer` | Too short or superficial | <5 words |

---

## Component Breakdown

### 1. Preprocessing Module (`preprocessing.py`)

**Purpose**: Normalize text for consistent analysis

**Functions**:
- `normalize_text()`: Lowercase + remove extra whitespace
- `tokenize()`: Split into words, preserve contractions
- `preprocess_pipeline()`: Combined preprocessing

**Example**:
```python
Input:  "Photosynthesis  is   IMPORTANT!"
Output: "photosynthesis is important"
Tokens: ["photosynthesis", "is", "important"]
```

### 2. Feature Extraction (`features.py`)

**Purpose**: Extract measurable characteristics from student answers

**Features Extracted**:
1. **Word Count**: Total words in answer
2. **Keyword Coverage**: Ratio of reference keywords found in student answer
3. **Spelling Errors**: Number of misspelled words using `pyspellchecker`

**Example**:
```python
Reference: "Photosynthesis uses sunlight, water, CO2 to make glucose"
Student:   "Plants use light to make food"

Features:
  - word_count: 6
  - keyword_coverage: 0.33 (33%)
  - num_spelling_errors: 0
  - missing_keywords: ["photosynthesis", "sunlight", "water", "glucose"]
```

### 3. Semantic Embeddings (`embeddings.py`)

**Purpose**: Understand semantic meaning beyond keywords

**How it works**:
1. Load Sentence-BERT model (`all-MiniLM-L6-v2`)
2. Convert text to 384-dimensional vector embedding
3. Compare embeddings using cosine similarity

**Why Sentence-BERT?**
- Captures semantic meaning (not just word matching)
- Distinguishes "plants use light" from "plants eat light"
- Fast and lightweight (80MB model)

**Example**:
```python
Reference: "Photosynthesis converts light into glucose"
Student A: "Plants change sunlight into sugar"      → Similarity: 0.78 (high)
Student B: "Plants eat sunlight to grow bigger"     → Similarity: 0.42 (low)
```

### 4. Error Detector (`detector.py`)

**Purpose**: Apply heuristic rules to classify errors

**Decision Logic** (in priority order):

```
IF similarity ≥ 0.75 AND keyword_coverage ≥ 0.5:
    → CORRECT (early exit with high confidence)

ELSE:
    Collect all issues:
    
    IF word_count < 5:
        → Flag as VAGUE_ANSWER
    
    IF spelling_errors ≥ 3:
        → Flag as LINGUISTIC_ERROR
    
    IF similarity < 0.55:
        → Flag as CONCEPTUAL_ERROR
    
    IF keyword_coverage < 0.4:
        → Flag as MISSING_KEY_CONCEPTS
    
    Return issue with HIGHEST confidence score
```

**Confidence Scoring**:
- Higher confidence for clearer violations
- Lower confidence for borderline cases
- Based on distance from thresholds

### 5. Optional Clustering Analysis (`analyzer.py`)

**Purpose**: Visualize if errors naturally group together

**Methods**:
- K-means clustering on embeddings
- t-SNE dimensionality reduction for 2D visualization
- Cluster purity analysis

---

## Dataset Structure

**Location**: `data/dataset.json`

**Format**:
```json
{
  "question_id": "Q1",
  "question_text": "Explain what photosynthesis is.",
  "reference_answer": "Photosynthesis is the process...",
  "student_answer": "Plants make food using light.",
  "is_correct": false,
  "error_label": "missing_key_concepts",
  "correct_answer_example": "Photosynthesis is the process where..."
}
```

**Statistics**:
- **Total samples**: 42 student answers
- **Questions**: 3 different questions (Q1: Photosynthesis, Q2: Water Cycle, Q3: Mitosis/Meiosis)
- **Distribution**:
  - Correct: 19%
  - Conceptual errors: 26%
  - Missing concepts: 24%
  - Linguistic errors: 17%
  - Vague answers: 14%

---

## Setup and Usage

### Installation

```bash
cd c:\ISEP\TESE\POC

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/Scripts/activate  # Git Bash
# OR
venv\Scripts\activate  # Windows CMD

# Install dependencies
pip install -r requirements.txt
```

### Run the POC

```bash
python main.py
```

**Expected Output**:
- Processing progress (10/42, 20/42, ...)
- Overall accuracy metrics
- Per-class precision/recall/F1
- Confusion matrix
- Sample predictions with explanations
- Results saved to `output/results.json`

### View Results

```bash
# Full results with all predictions
cat output/results.json

# Visualization (if running clustering)
open output/cluster_visualization.png
```

---

## Project Structure

```
POC/
├── data/
│   └── dataset.json           # 42 synthetic student answers
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Text normalization & tokenization
│   ├── features.py            # Keyword extraction & spelling check
│   ├── embeddings.py          # Sentence-BERT similarity
│   ├── detector.py            # Heuristic error detection rules
│   └── analyzer.py            # Optional clustering analysis
│
├── output/
│   └── results.json           # Prediction results & metrics
│
├── venv/                      # Virtual environment (created on setup)
├── main.py                    # Main pipeline runner
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | ≥2.2.0 | Semantic embeddings (Sentence-BERT) |
| `pyspellchecker` | ≥0.7.0 | Spelling error detection |
| `scikit-learn` | ≥1.0.0 | Clustering & metrics |
| `numpy` | ≥1.21.0 | Numerical operations |
| `matplotlib` | ≥3.5.0 | Visualization |

**Total size**: ~200MB (includes PyTorch for Sentence-BERT)

---

## Performance Analysis

### Strengths

✅ **Perfect detection** (100% precision & recall):
- Linguistic errors: All spelling-heavy errors caught
- Vague answers: All too-short answers caught

✅ **High precision** (100%):
- Correct answers: When predicted as correct, always accurate

### Limitations

⚠️ **Conceptual error detection** (9% recall):
- Often confused with "missing key concepts"
- Challenge: Both have low keyword coverage
- Requires deeper semantic contradiction detection

⚠️ **Dataset**:
- Small (42 samples)
- Synthetic (not real student data)
- English-only

### Future Improvements

1. **Better conceptual error detection**:
   - Add textual entailment models
   - Detect semantic contradictions
   - Use phrase-level embedding comparison

2. **Dataset expansion**:
   - Real student answers from classrooms
   - Multi-language support
   - More question types (math, history, etc.)

3. **Hybrid approach**:
   - Combine heuristics with ML classifiers
   - Train on larger labeled dataset
   - Fine-tune Sentence-BERT on educational data

---

## Scientific Value

This POC successfully demonstrates:

1. ✅ **Error patterns exist**: Different error types have distinct characteristics
2. ✅ **Automated detection is viable**: 66.7% accuracy with simple heuristics
3. ✅ **Interpretability**: Each prediction includes human-readable explanation
4. ✅ **Efficiency**: Processes 42 answers in ~10 seconds on CPU
5. ✅ **Scalability**: Modular architecture allows easy extension

**Thesis Contribution**:
- Proves AI-assisted educational feedback is technically feasible
- Identifies clear research challenges (conceptual vs. missing concepts)
- Provides baseline for future improvements
- Demonstrates language-agnostic potential

---

## Citation

If you use this POC in your research, please cite:

```
Error Pattern Detection in Student Text Answers
Proof of Concept for AI-Assisted Educational Feedback
[Your Name], [Institution], 2026
```

---

## Limitations (To Discuss in Thesis)

1. **Synthetic Dataset**: Not validated against real classroom data
2. **Simplified Labels**: Ground truth may not reflect pedagogical nuance
3. **English Focus**: Single language, lacks multilingual validation
4. **Threshold Sensitivity**: Heuristic values may need domain-specific tuning
5. **Conceptual Error Confusion**: Hardest distinction for automated systems

These limitations are **expected for a POC** and provide valuable direction for future research.

---

## License

Academic research project. For educational purposes only.
