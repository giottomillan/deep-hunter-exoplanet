# 🪐 Deep Hunter Exoplanet

Deep Hunter Exoplanet is a deep learning research project focused on detecting exoplanet transit candidates from astronomical light curves.

The project is inspired by **VARnet**, a signal-processing neural architecture developed for large-scale astronomical time-series classification, combining convolutional neural networks with Fourier and wavelet-based feature extraction.

The goal is to explore whether similar techniques can improve the detection of weak planetary transit signals in Kepler and TESS observations.

---

## 🌌 Motivation

Thousands of exoplanets have been discovered using the transit method, where a planet crossing in front of its host star produces a small periodic decrease in observed brightness.

Detecting these signals is challenging because:

- Transit depth can be extremely small
- Stellar activity introduces noise
- Instrumental effects distort measurements
- False positives such as eclipsing binaries can mimic planets

This project investigates deep learning approaches capable of identifying these subtle patterns directly from light curves.

---

## 🔭 Data Sources

Initial development focuses on:

### Kepler Mission

- High-quality long-duration light curves
- Reliable confirmed planet labels
- Ideal benchmark dataset for model development

Future extensions:

- TESS full-sky survey
- Cross-mission generalization

---

## 🧠 Approach

Pipeline:
Kepler Light Curve

    ↓

Preprocessing

Cleaning
Normalization

Detrending

  ↓

Feature Extraction

CNN temporal features
Fourier representation

Wavelet decomposition

  ↓

Deep Learning Classifier

    ↓

Exoplanet Transit Probability


---

## 🏗️ Planned Architecture

### Baseline Model

1D Convolutional Neural Network:


Flux Time Series

↓
Conv1D

↓
Pooling

↓
Dense Layers

↓

Planet / Non Planet


### Advanced Model: ExoVARnet

Inspired by VARnet:

- Temporal CNN branch
- Fourier feature extraction
- Wavelet denoising branch
- Feature fusion classifier

---

## 🎯 Objectives

- [ ] Download and process Kepler light curves
- [ ] Build labeled exoplanet dataset
- [ ] Train CNN baseline classifier
- [ ] Evaluate precision, recall and F1-score
- [ ] Implement Fourier-based features
- [ ] Add wavelet signal processing
- [ ] Compare against baseline methods
- [ ] Extend experiments to TESS

---

## 📊 Evaluation Metrics

Primary metrics:

- Precision
- Recall
- F1-score
- ROC-AUC

Special focus:

- Minimize false negatives
- Detect weak transit signals

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Lightkurve
- Astropy
- NumPy
- SciPy
- PyWavelets
- Scikit-learn
- Matplotlib

---

## 🚀 Long Term Vision

Create a scalable AI system capable of analyzing astronomical surveys and identifying potential undiscovered exoplanet candidates.

Future improvements:

- Multi-class stellar variability classification
- Explainable AI for transit detection
- Self-supervised learning on astronomical time-series
- Deployment on large-scale sky surveys

---

## References

- Paz, M. (2024). VARnet: A Submillisecond Fourier and Wavelet-based Model to Extract Variable Candidates from the NEOWISE Single-exposure Database.
- NASA Kepler Mission
- NASA TESS Mission
