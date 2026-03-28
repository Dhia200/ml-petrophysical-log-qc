# Machine Learning-Assisted Petrophysical Log Quality Control

Automated anomaly detection, electrofacies classification, and washout reconstruction using unsupervised and supervised machine learning algorithms.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Dhia200-181717?logo=github)](https://github.com/Dhia200/ml-petrophysical-log-qc)

---

## 📖 Overview

Borehole washouts corrupt up to **70% of raw well log data**, requiring manual quality control that consumes 50-70% of petrophysical workflow time. This project demonstrates a fully automated, edge-computing-ready machine learning pipeline that:

- **Detects anomalies** using multi-dimensional Local Outlier Factor (LOF) analysis
- **Classifies rock types** with probabilistic Gaussian Mixture Models (GMM)
- **Reconstructs corrupted logs** using facies-specific Random Forest regression

### Key Results

| Metric | Value | Impact |
|--------|-------|--------|
| **Accuracy (R²)** | 0.873 - 0.876 | 87% of density variance explained |
| **Precision (RMSE)** | 0.034 - 0.041 g/cc | <1.5% porosity calculation error |
| **Speed** | 2 days | Previously 15 days manual QC (7.5x faster) |
| **Validation** | 2 fields | Norway + Netherlands |

---

## 🎯 Problem Statement

During drilling operations, borehole enlargement (washouts) causes pad-contact tools like bulk density sensors to lose formation contact and measure drilling mud instead of rock properties. Traditional workflows require tedious manual quality control and subjective corrections.

### Industry Impact

- **70%** of raw well log data initially flagged as unreliable
- **50-70%** of petrophysicist time consumed by manual QC
- Inconsistent data harmonization across multi-well projects
- Operational delays of 10-15 days per well

---

## 🔬 Methodology

### Workflow Architecture

```
Raw Well Logs (LAS)
        ↓
[1] Washout Detection (Caliper-based)
        ↓
[2] Anomaly Detection (LOF + PCA)
        ↓
[3] Electrofacies Classification (GMM)
        ↓
[4] Facies-Specific Reconstruction (Random Forest)
        ↓
Quality-Controlled Density Log
```

### Phase 1: Automated Washout Detection

- **Algorithm:** Caliper-based threshold detection
- **Threshold:** Δ Caliper > 1.0 inch
- **Output:** Binary washout flag per depth point

### Phase 2: Multi-Dimensional Anomaly Detection

- **Algorithm:** Local Outlier Factor (LOF)
- **Parameters:** n_neighbors=20, contamination=0.05
- **Visualization:** Principal Component Analysis (PCA) for 2D projection
- **Advantage:** Detects tool malfunctions even when caliper appears normal

### Phase 3: Electrofacies Classification

- **K-Means Clustering:** Initial rock typing (baseline)
- **Gaussian Mixture Models (GMM):** Probabilistic classification
- **Critical Innovation:** Uses **only robust logs** (GR, AC) to avoid corrupted measurements
- **Output:** 3 distinct lithofacies (Clean Sand, Mixed, Shale)

### Phase 4: Facies-Specific Log Reconstruction

- **Algorithm:** Random Forest Regression
- **Parameters:** n_estimators=100, max_depth=10
- **Architecture:** Three independent models per lithology
- **Training:** 80% clean data, 20% blind testing
- **Routing:** Each washout reconstructed by its corresponding facies model

---

## 📊 Results

### Volve Field (Norway) - Primary Dataset

**Dataset:** Equinor Volve Open Data, Well 15/9-19  
**Logs Used:** GR, CALI, AC, DEN, NEU, RDEP

| Metric | Value |
|--------|-------|
| R² Score | 0.873 |
| RMSE | 0.034 g/cc |
| Porosity Error | <1.5% |
| QC Timeline | 2 days (vs 15 days manual) |

### Netherlands Field - Validation Dataset

**Adaptations:**
- No caliper log → Used DRHO (density correction) for washout detection
- No resistivity → Trained Random Forest with only 3 features (GR, AC, NEU)

| Metric | Value |
|--------|-------|
| R² Score | 0.876 |
| RMSE | 0.041 g/cc |
| Generalization | Maintained accuracy despite different log suite |

**Key Insight:** Consistent performance across different fields and logging configurations validates the workflow's generalizability.

---

## 📁 Repository Structure

```
ml-petrophysical-log-qc/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/
│   └── README.md                     # Dataset information
│
├── src/
│   └── petrophysical_qc_pipeline.py  # Main Python script (488 lines)
│
├── results/
│   ├── 01_washout_detection.png      # Post 1: Caliper-based QC
│   ├── 02_random_forest_reconstruction.png
│   ├── 03_kmeans_classification.png
│   ├── 04_lof_pca_anomaly.png
│   ├── 05_gmm_crossplot.png
│   └── 06_facies_specific_final.png  # Final reconstruction
│
├── docs/
│   └── technical_paper.pdf           # Detailed methodology (optional)
│
└── notebooks/
    └── (Jupyter notebook - optional)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Dhia200/ml-petrophysical-log-qc.git
cd ml-petrophysical-log-qc

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python src/petrophysical_qc_pipeline.py
```

The script will:
1. Load data from Equinor Volve Field (online)
2. Execute all 6 workflow stages
3. Generate visualization outputs
4. Display performance metrics

---

## 📦 Dependencies

**Core Requirements:**
```
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
scikit-learn >= 1.0.0
lasio >= 0.27
```

**Installation:**
```bash
pip install -r requirements.txt
```

**Python Version:** 3.8 or higher

---

## 📈 Visualizations

### Post 1: Automated Washout Detection
![Washout Detection](results/01_washout_detection.png)
*Caliper-based identification of borehole enlargement zones with automatic flagging*

### Post 2: AI Log Reconstruction
![Random Forest Reconstruction](results/02_random_forest_reconstruction.png)
*ML-reconstructed density (blue) vs corrupted original data (red dashed)*

### Post 3: K-Means Rock Classification
![K-Means Clustering](results/03_kmeans_classification.png)
*Automated electrofacies identification using unsupervised clustering*

### Post 4: Multi-Dimensional Anomaly Detection
![LOF + PCA](results/04_lof_pca_anomaly.png)
*PCA visualization showing separation of valid data (blue) from anomalies (red)*

### Post 5: Gaussian Mixture Models
![GMM Classification](results/05_gmm_crossplot.png)
*Probabilistic clustering on GR vs Acoustic cross-plot*

### Post 6: Facies-Specific Reconstruction (Final Result)
![Final Reconstruction](results/06_facies_specific_final.png)
*Complete pipeline: Detection → Classification → Reconstruction*

---

## 🛠️ Technical Details

### Algorithms & Hyperparameters

| Stage | Algorithm | Key Parameters |
|-------|-----------|----------------|
| Anomaly Detection | Local Outlier Factor | n_neighbors=20, contamination=0.05 |
| Dimensionality Reduction | PCA | n_components=2 |
| Clustering | Gaussian Mixture Model | n_components=3, covariance='full' |
| Reconstruction | Random Forest Regression | n_estimators=100, max_depth=10 |

### Feature Engineering

**Washout Detection:**
```python
washout_flag = (CALI - BIT_SIZE) > 1.0  # inch
```

**GMM Input (Robust Logs Only):**
- Gamma Ray (GR)
- Acoustic Slowness (AC)

**Random Forest Input (All Available Logs):**
- GR, AC, NEU, RDEP (for density prediction)

### Model Training Strategy

- **Train/Test Split:** 80% / 20%
- **Facies-Specific Models:** Three independent Random Forest regressors
- **Validation:** Blind testing on holdout data
- **Generalization:** Cross-field validation (Norway → Netherlands)

---

## 📄 Dataset

**Primary Source:** [Equinor Volve Field Open Data](https://www.equinor.com/energy/volve-data-sharing)

**Details:**
- **Well:** 15/9-19
- **Location:** North Sea, Norway
- **Format:** LAS (Log ASCII Standard)
- **Logs:** GR, CALI, DEN, NEU, AC, RDEP
- **Depth Range:** ~3000-4000 meters
- **License:** Creative Commons (CC BY-NC-SA 4.0)

**Data Access:**
The pipeline automatically downloads data from:
```
https://raw.githubusercontent.com/andymcdgeo/Petrophysics-Python-Series/master/Data/15-9-19_SR_COMP.LAS
```

**Attribution:**  
All data remains property of Equinor ASA and is used in accordance with Creative Commons license terms.

---

## 🎓 Citation

If you use this work in your research or projects, please cite:

```bibtex
@software{alioua2026_petrophysical_qc,
  author = {Alioua, Dhia Eddine},
  title = {Machine Learning-Assisted Petrophysical Log Quality Control: 
           Automated Anomaly Detection and Washout Reconstruction},
  year = {2026},
  url = {https://github.com/Dhia200/ml-petrophysical-log-qc},
  note = {Validated on Equinor Volve Field (Norway) and Netherlands datasets}
}
```

**Technical Paper:** See `docs/technical_paper.pdf` for detailed methodology and validation results.

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary:**
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Must include license and copyright notice

---

## 👤 Author

**Dhia Eddine Alioua**  
Petroleum Geophysicist | Machine Learning Enthusiast

📧 **Email:** alioua.dhia@gmail.com  
🔗 **LinkedIn:** [linkedin.com/in/dhia-eddine-alioua](https://www.linkedin.com/in/dhia-eddine-alioua)  
💻 **GitHub:** [github.com/Dhia200](https://github.com/Dhia200)  

---

## 🙏 Acknowledgments

- **Equinor ASA** for open-sourcing the Volve Field dataset
- **Andy McDonald** for curating and hosting petrophysics Python resources
- **scikit-learn community** for robust, well-documented ML implementations
- **Open-source contributors** who made this workflow possible

---

## 📚 References

1. **Serra, O.** (1984). *Fundamentals of Well-Log Interpretation*. Elsevier.
2. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
3. **Breunig, M. M., et al.** (2000). LOF: Identifying Density-Based Local Outliers. *ACM SIGMOD Record*, 29(2), 93-104.
4. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## 🔮 Future Work

- [ ] Real-time edge deployment on rig-side hardware
- [ ] Integration with additional log types (image logs, NMR, elemental spectroscopy)
- [ ] Extension to carbonate reservoirs
- [ ] API development for integration with commercial software (Techlog, Petrel)
- [ ] Uncertainty quantification for reconstructed logs
- [ ] Multi-well field-scale harmonization

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

**How to contribute:**
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⭐ Star This Repository

If you found this project useful, please consider giving it a star! It helps others discover the work and motivates continued development.

[![GitHub stars](https://img.shields.io/github/stars/Dhia200/ml-petrophysical-log-qc?style=social)](https://github.com/Dhia200/ml-petrophysical-log-qc/stargazers)

---

## 📞 Questions or Feedback?

- **Open an issue** on GitHub for bugs or feature requests
- **Email me directly** at alioua.dhia@gmail.com
- **Connect on LinkedIn** for professional inquiries

---

**Last Updated:** March 2026  
**Status:** ✅ Production-ready | 📖 Fully documented | 🧪 Validated on 2 fields
