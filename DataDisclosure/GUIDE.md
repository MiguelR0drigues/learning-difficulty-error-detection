# Systematic Review on Data Disclosure in AI/ML Systems
## Research Coordination Document

### Overview
This document serves as a coordination guide for executing the systematic review on data disclosure in AI/ML systems using PRISMA 2020 guidelines. It complements the LaTeX template provided and gives actionable steps for study selection, data extraction, and refinement.

---

## Phase 1: Search Strategy & Record Collection

### Search Strings (Final Version)

**Search 1: General Data Disclosure & Privacy Attacks**
```
("data disclosure" OR "privacy leakage" OR "membership inference" OR "model inversion" OR "training data extraction") 
AND ("machine learning" OR "deep learning" OR "neural network" OR "large language model") 
AND ("attack" OR "defense" OR "privacy-preserving" OR "privacy attack")
```

**Search 2: Differential Privacy & Federated Learning**
```
("differential privacy" OR "federated learning" OR "secure aggregation" OR "homomorphic encryption" OR "secure multi-party computation") 
AND ("machine learning" OR "neural network" OR "deep learning") 
AND ("empirical" OR "evaluation" OR "experiment")
```

**Search 3: Privacy Attacks on LLMs & Specific Models**
```
("training data extraction" OR "prompt injection" OR "privacy attack" OR "membership inference") 
AND ("LLM" OR "language model" OR "ChatGPT" OR "GPT" OR "transformer") 
AND (2023:2026)
```

**Search 4: Educational & Domain-Specific Privacy**
```
("privacy" OR "data disclosure" OR "anonymization") 
AND ("machine learning" OR "AI" OR "predictive model") 
AND ("education" OR "healthcare" OR "finance" OR "sensitive data")
```

### Filters to Apply (All Searches)
- **Year**: 2018–2026
- **Language**: English
- **Document type**: Peer-reviewed (journal or conference)
- **Exclude**: theses, technical reports, white papers, opinion pieces

### Expected Yield
- Search 1: 80–120 records
- Search 2: 60–100 records
- Search 3: 40–60 records
- Search 4: 50–80 records
- **Total (with duplicates)**: ~200–350 records
- **After deduplication**: ~120–200 candidate studies

---

## Phase 2: Title & Abstract Screening (Triage)

### Screening Criteria Quick Reference

**INCLUDE if**:
- Explicitly evaluates privacy in ML (attack or defense)
- Empirical study with data/experiments
- ML/DL model is central (not incidental)
- Published 2018–2026, English, peer-reviewed

**EXCLUDE if**:
- Pure security/cryptography (no ML component)
- Pure governance/policy (no technical evaluation)
- Opinion/position papers without empirical data
- Reviews or meta-analyses (include separately if insights are actionable)
- Not in English

**UNCLEAR → Flag for Full-Text Review**

### Screening Process
1. Create a **Google Sheet** with columns: `RecordID | Authors | Title | Year | Abstract | InitialScreening (Include/Exclude/Unclear) | Reason | Notes`
2. One reviewer screens all titles/abstracts
3. Mark screening date and reviewer initials
4. **Expected outcome**: ~40–60 records move to full-text review

---

## Phase 3: Full-Text Review & Data Extraction

### Data Extraction Template (Google Sheets Columns)

Create a second sheet with these columns:

| Field | Description | Example |
|-------|-------------|---------|
| `StudyID` | Unique identifier | SR001, SR002, ... |
| `Authors` | First author (et al.) | Shokri et al. |
| `Year` | Publication year | 2021 |
| `Title` | Full title | "Membership Inference Attacks..." |
| `Venue` | Journal/Conference & year | ICML 2021 |
| `URL_DOI` | Link to paper | https://arxiv.org/... |
| `ThreatType` | Primary threat | membership_inference / model_inversion / extraction / poisoning / evasion |
| `DefenseType` | Primary defense | dp_sgd / federated_learning / smpc / encryption / distillation / anonymization / other |
| `AIMLTechnique` | Model(s) studied | logistic_regression, random_forest, cnn, lstm, transformer, gnn, other |
| `DatasetName` | Dataset(s) used | MNIST, CIFAR-10, synthetic, healthcare_real, education_real |
| `DatasetSize` | No. of samples / records | e.g., "60,000 training" |
| `ApplicationDomain` | Context | general_ml, healthcare, finance, education, cybersecurity, other |
| `AttackSuccessRate` | Metric for attack efficacy | e.g., "91% AUC" or "ε=3, δ=1e-5" |
| `UtilityMetric` | Model performance with defense | e.g., "Accuracy: 92% (baseline 95%)" |
| `PrivacyMetric` | Quantitative privacy claim | epsilon/delta, attack precision, privacy bound, or qualitative |
| `UtilityLoss` | % degradation | e.g., "-3%" or "15% accuracy drop" |
| `ReproducibilityScore` | Code/data available? | full_reproducible / partial / not_reproducible |
| `RiskOfBias` | Low / Moderate / High | Based on checklist |
| `KeyFindings` | 1–2 sentence summary | "DP-SGD achieves privacy at 10% accuracy cost on CIFAR-10" |
| `LimitationsNoted` | Study limitations | "Synthetic data only", "Limited scaling", "No comparison to DP" |
| `EducationRelevance` | Applicable to EdTech? | Yes / No / Partial |
| `RelevantToReview` | Include in final synthesis? | Yes / No |
| `ReviewerNotes` | Extraction notes | Free text |

### Risk of Bias Checklist

For each included study, score (✓ = criterion met, ✗ = not met):

- [ ] **Threat model clearly defined** (attacker knowledge & capabilities explicit)
- [ ] **Multiple baselines or comparisons** (not evaluated against only one defense/attack)
- [ ] **Realistic or diverse dataset** (not synthetic-only on toy data)
- [ ] **Reproducibility** (code, data, or sufficient implementation details provided)
- [ ] **Statistical reporting** (confidence intervals, multiple runs, or significance tests)

**Risk Classification**:
- **Low**: ≥4 criteria met
- **Moderate**: 2–3 criteria met
- **High**: ≤1 criterion met

---

## Phase 4: Synthesis & Reporting

### Organization by Threat Type

#### Membership Inference
- Sub-organize by: attack model (black-box vs. white-box), dataset characteristics, model architecture
- Report: typical success rates, factors increasing vulnerability, defenses applied

#### Model Inversion
- Sub-organize by: input dimensionality (image, tabular, text), domain (vision, NLP, general)
- Report: reconstruction quality metrics, feasibility of full vs. partial reconstruction

#### Training Data Extraction
- Sub-organize by: model type (LLM vs. classical), attack method (prompt-based, gradient-based)
- Report: extraction rates, data memorization effects, model size/training set size ratios

#### Poisoning / Evasion
- Sub-organize by: attack scenario (training-time vs. inference-time)
- Report: success metrics, mitigation via adversarial training or robust learning

### Organization by Defense Type

#### Differential Privacy
- Sub-organize by: mechanism (DP-SGD, DP-FL, DP-inference), algorithm (Gaussian noise, Laplace)
- Report: typical ε values used, corresponding accuracy loss, practical feasibility

#### Federated Learning
- Sub-organize by: federation architecture (horizontal, vertical), additional privacy layers (DP, SMPC)
- Report: communication overhead, data heterogeneity challenges, privacy-utility profiles

#### Cryptographic (SMPC, HE)
- Sub-organize by: cryptographic primitive, application (aggregation, prediction)
- Report: computational cost, security model, practical deployment examples

#### Anonymization / Pseudonymization
- Sub-organize by: technique (k-anonymity, l-diversity, synthetic data, perturbation)
- Report: re-identification risk, information loss, suitability for downstream ML

#### Other (Distillation, Adversarial Training, Filtering, Detection)
- Report separately with brief summary

### Expected Table for Final Write-Up

Create **Table 1** (Study Characteristics) in LaTeX with representative studies (40–50 total):

| Threat | Defense | Study | Model(s) | Privacy Metric | Utility Metric | Key Trade-Off |
|--------|---------|-------|----------|----------------|----------------|---------------|
| MI | DP | Author et al. (2021) | NN | AUC 0.65 | Acc 89% | ε=3 → 5% loss |
| MI | FL | Author et al. (2022) | RF | Attack succ. 30% | Acc 92% | Comm. overhead 10× |
| Inversion | HE | Author et al. (2023) | CNN | SSIM 0.4 | Time 50s/pred | Inference latency |
| ... | ... | ... | ... | ... | ... | ... |

---

## Phase 5: Refinement & Quality Checks

### Pre-Publication Checklist

- [ ] All included studies are peer-reviewed and from 2018–2026
- [ ] PICOS criteria applied consistently
- [ ] Data extraction template complete for all studies
- [ ] Risk of bias scores assigned
- [ ] PRISMA 2020 checklist completed
- [ ] Quantitative synthesis (if any) uses consistent metrics
- [ ] Qualitative synthesis organized logically (by threat type, defense, domain)
- [ ] No data extraction conflicts or LLM errors in summaries
- [ ] Tables populated with representative studies
- [ ] Conclusions supported by evidence in results section
- [ ] Limitations section acknowledges gaps and methodological issues
- [ ] References complete and formatted correctly

### Key Metrics to Report

1. **Study yield**: Total identified, screened, included, flow diagram
2. **Study characteristics**: Distribution by year, domain, threat type, defense type
3. **Threat landscape**: Frequency of each attack type, typical success rates
4. **Defense effectiveness**: Privacy-utility trade-offs, computational costs
5. **Methodological quality**: Risk of bias distribution, reproducibility assessment
6. **Domain specificity**: Proportion of educational studies, healthcare vs. finance, etc.

### Final Word Count Budget (8 Pages)

- **Title & Abstract**: ~250 words
- **Introduction**: ~1000 words
- **Methods**: ~1200 words (PICOS, search, selection, extraction, bias assessment)
- **Results**: ~2000 words (characteristics, threat landscape, defenses, risk of bias)
- **Discussion**: ~1500 words (synthesis, implications, gaps, future work)
- **Conclusion**: ~300 words
- **References**: ~500 words (estimate for ~50–60 citations)
- **Total**: ~7000–8000 words

---

## Tools & Infrastructure

### Google Sheets Template
- **Sheet 1**: Records (search results, screening decisions)
- **Sheet 2**: Included studies (full data extraction)
- **Sheet 3**: Risk of bias summary
- **Sheet 4**: Synthesis notes (by threat/defense/domain)

### LaTeX Compilation
- Use `pdflatex` or `xelatex` for final PDF
- Ensure bibliography file (`.bib`) is populated
- Adjust margins/font size if needed to fit 8 pages

### Collaboration
- Share Google Sheets with reviewers (if second human reviewer joins later)
- Use version control (Git/GitHub) for LaTeX document
- Maintain a shared "notes" file for methodological decisions

---

## Timeline & Milestones

| Phase | Task | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | Searches, deduplication | 1–2 weeks | Records compiled in Sheets |
| 2 | Title/abstract screening | 1–2 weeks | 40–60 candidates identified |
| 3 | Full-text review & extraction | 3–4 weeks | Data extraction complete, risk of bias assigned |
| 4 | Synthesis & writing | 2–3 weeks | Results & discussion drafted |
| 5 | Refinement, proofreading | 1 week | Final LaTeX version ready |

---

## Key Contacts & Decision-Making

### For Borderline Inclusion Decisions

- **Criterion**: If threat type or defense is clear but domain is unexpected → INCLUDE
- **Criterion**: If study lacks quantitative evaluation but proposes framework with proof-of-concept → INCLUDE if empirical validation present, EXCLUDE if purely theoretical
- **Criterion**: If study combines multiple domains (e.g., healthcare + finance) → INCLUDE and note in "Domain" column as "multi-domain"

### For Disagreements on Risk of Bias

- Default to **Moderate** if criteria are mixed
- Document reasoning in `ReviewerNotes`

---

## References for Methodological Standards

1. PRISMA 2020: https://www.prisma-statement.org/
2. PRISMA-trAIce (for AI-assisted reviews): https://ai.jmir.org/2025/1/e80247
3. Cochrane Risk of Bias Tool (adapted for ML/security): https://training.cochrane.org/
4. Systematic review guidance (computer science): https://dlnext.acm.org/

---

## Next Steps (When Ready to Execute)

1. ✅ Finalize and validate search strings (test in 1–2 databases first)
2. ✅ Create Google Sheets template and share access
3. ✅ Execute searches and populate Sheet 1 (Records)
4. ✅ Begin title/abstract screening; aim for ~50 candidates
5. ✅ Retrieve full texts (use institutional access, ResearchGate, or author contact)
6. ✅ Full-text screening and data extraction (Sheet 2)
7. ✅ Risk of bias assessment (Sheet 3)
8. ✅ Synthesis writing (Sheet 4 informs narrative)
9. ✅ Populate LaTeX template with results
10. ✅ Final review, proofs, citation check
11. ✅ Submit or publish

---

Good luck with the review! This is a high-impact topic, and a well-executed PRISMA-compliant review will be a valuable contribution to the field.