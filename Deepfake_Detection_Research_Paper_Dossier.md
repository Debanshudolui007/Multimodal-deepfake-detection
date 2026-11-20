# Research Paper Dossier: Tri-Modal Cross-Attention Fusion for Robust Deepfake Detection

---

## 1. Paper Identification & Metadata

* **Primary Proposed Title:** *Tri-Modal Cross-Attention Fusion: Harmonizing Spatial, Spectral, and Acoustic Representations for Robust Deepfake Detection*
* **Alternative Working Titles:**
  * *Beyond Visual Artifacts: A Spatial-Frequency-Audio Multimodal Framework with Attention-Guided Forgery Attribution*
  * *Swin-Transformed Spectral and BiLSTM-Acoustic Synergy for Multimodal Deepfake Video Detection*
* **Author Affiliation / Project Context:** Advanced Multimedia Forensics & Deep Learning Project
* **Keywords:** Deepfake Detection, Multimodal Fusion, Cross-Attention Transformer, Discrete Wavelet Transform (DWT), 2D Fast Fourier Transform (FFT), Swin Transformer, Bidirectional LSTM, Platt Scaling, Explainable AI (XAI), Media Forensics.
* **Target Publication Venues:** IEEE Transactions on Information Forensics and Security (TIFS), IEEE CVPR / ICCV / ECCV Workshops on Media Forensics, ACM Multimedia (ACM MM), IEEE ICASSP, Elsevier Computers & Security.

---

## 2. Abstract

Recent advances in Generative Adversarial Networks (GANs) and diffusion models have made deepfake video and audio synthesis increasingly indistinguishable from authentic media. Traditional unimodal detection pipelines rely primarily on visual artifacts in the spatial domain, which degrade significantly under lossy social media compression, resizing, and adversarial perturbations. In this work, we present an end-to-end tri-modal deepfake detection architecture that jointly models **Spatial (RGB)**, **Frequency (FFT + DWT)**, and **Acoustic (MFCC + Mel-Spectrogram)** representations.

Our pipeline extracts facial keyframes via RetinaFace and constructs a composite 3-channel frequency representation combining 2D FFT log-magnitude, Daubechies-4 Wavelet (DWT) Low-Low approximation, and high-frequency Wavelet Energy maps ($\sqrt{LH^2 + HL^2 + HH^2}$). For temporal audio, we extract concatenated 40-dimensional MFCCs and 128-dimensional Mel spectrograms across a fixed temporal window. The system leverages specialized backbones: a **ResNeXt-50 (32×4d)** for spatial feature extraction (2048-D), a **Swin Transformer Tiny** with shifted-window self-attention for spectral lattice artifact discovery (768-D), and a **1D CNN + BiLSTM with temporal attention pooling** for phonemic and vocal anomaly modeling (512-D). These representations are projected into a shared 512-D latent space and fused via a **Multi-Head Transformer Cross-Attention module** with dynamic attention pooling and modality dropout (10%).

Trained and validated across **35,089 videos (34.7 GB)** from three standard benchmarks (**Celeb-DF v2, FaceForensics++ C23, and FakeAVCeleb v1.2**), our individual branches achieve validation ROC-AUCs of **96.76% (Spatial)**, **97.03% (Frequency)**, and **100.0% (Audio)**, with the unified cross-attention fusion achieving **95.68% ROC-AUC, 94.19% accuracy, and 96.69% F1-score**. Post-hoc Platt scaling is incorporated for well-calibrated posterior probabilities, while Grad-CAM and modality-level attention weight breakdowns provide human-interpretable evidence attribution.

---

## 3. Introduction & Research Motivation

### 3.1 The Deepfake Challenge
The democratization of generative AI models (such as StyleGAN, SimSwap, Wav2Lip, and latent diffusion architectures) has enabled photorealistic face swapping, facial reenactment, and voice cloning. Malicious use cases—such as political disinformation, financial fraud, impersonation, and non-consensual imagery—pose profound threats to digital information ecosystems.

### 3.2 Vulnerability of Unimodal Detection
1. **Visual Degradation in Spatial Domain:** Modern synthesizers generate seamless blending boundaries. Furthermore, social media platforms re-compress videos using lossy codecs (H.264/HEVC), wiping out subtle spatial inconsistencies.
2. **Spectral Footprints:** Convolutional upsampling (transposed convolutions, nearest-neighbor interpolation) introduces characteristic periodic lattice artifacts and high-frequency energy discrepancies in the frequency domain. While standard CNNs with localized receptive fields often miss global spectral correlations, self-attention architectures operating over frequency decompositions can capture these artifacts.
3. **Cross-Modal Inconsistencies:** Modern deepfakes frequently manipulate only one modality (e.g., pristine real video with voice-cloned audio, or authentic audio paired with a lip-synced face). Single-modality detectors fail when the forgery exists outside their observation domain.

### 3.3 Core Research Contributions
1. **Tri-Modal Synergy:** We formulate an end-to-end framework integrating Spatial, Frequency, and Acoustic domains, explicitly tackling visual, spectral, and auditory manipulation.
2. **Engineered Frequency Tensor with Swin Attention:** We construct a 3-channel frequency representation combining 2D FFT log-magnitude, DWT approximation subband, and DWT energy maps, processed via a Swin Transformer Tiny to model long-range frequency correlations.
3. **Temporal Acoustic Modeling:** A hybrid 1D CNN + BiLSTM network with temporal self-attention pooling captures phoneme-level and utterance-level synthetic speech artifacts.
4. **Cross-Attention Fusion with Modality Robustness:** A Multi-Head Transformer Encoder enables inter-modality reasoning. Modality Dropout and Boolean Modality Masking prevent the model from learning missingness as a spurious shortcut.
5. **Post-Hoc Probability Calibration & Explainability:** Learned Platt scaling calibrates output logits to reliable posterior probabilities, complemented by Grad-CAM and dynamic modality contribution visualizations.

---

## 4. Benchmark Datasets & Preprocessing Protocol

### 4.1 Dataset Corpus

The complete system was developed, trained, and tested across three widely recognized forensic benchmarks totaling **35,089 videos (34.69 GB)**:

| Dataset | Total Videos | Real Count | Fake Count | Forgery Classes / Techniques | Codecs & Specs |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Celeb-DF (v2)** | 6,529 | 890 | 5,639 | High-quality Deepfake face swap | FMP4 / MPEG-4, ~30 fps |
| **FaceForensics++ (FF++ C23)** | 7,000 | 1,000 | 6,000 | Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, DeepFakeDetection | H.264 (compressed C23), ~27 fps |
| **FakeAVCeleb (v1.2)** | 21,560 | 500 | 21,060 | FakeVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-FakeAudio (Wav2Lip, SV2TTS) | H.264 & FMP4, 224×224, 25 fps |
| **Combined Corpus** | **35,089** | **2,390** | **32,699** | **Tri-modal visual and acoustic forgeries** | **34.69 GB Total** |

### 4.2 Data Partitioning Integrity
* **Split Ratios:** 70% Training, 15% Validation, 15% Test.
* **Integrity Guard:** Partitioning is performed strictly at the **video ID and identity level**, ensuring that all frames and audio segments from any given identity/video remain strictly in a single split to eliminate data leakage.

### 4.3 Preprocessing & Feature Engineering
1. **Face Localization & Normalization:**
   * Keyframe sampling: frames are extracted and processed with **RetinaFace** to detect the primary facial bounding box.
   * Facial crops are aligned and scaled to $224 \times 224 \times 3$.
   * Quality filtering: Laplacian blur score variance threshold ($> 30.0$) and luminance thresholding reject corrupted frames.
2. **Composite Frequency Tensor Generation ($224 \times 224 \times 3$):**
   * Grayscale conversion: $I_{gray} \in [0, 1]$.
   * **Channel 0 (FFT Log-Magnitude Spectrum):** Centered 2D Fast Fourier Transform:
     $$F(u, v) = \log(1 + |\text{fftshift}(\mathcal{F}(I_{gray}))|)$$
     Captures global spectral grid artifacts from generative upsamplers.
   * **Channel 1 (DWT Low-Low Approximation):** 2D Discrete Wavelet Transform with Daubechies-4 (`db4`) filter at level 1:
     $$\text{LL} = \text{DWT}_{db4}(I_{gray})$$
     Preserves core structural approximations.
   * **Channel 2 (DWT High-Frequency Energy Map):** Detail subbands (Horizontal $LH$, Vertical $HL$, Diagonal $HH$):
     $$E_{dwt} = \sqrt{LH^2 + HL^2 + HH^2}$$
     Highlights high-frequency edge and blending anomalies.
   * Standardized via global dataset statistics: $\mu = (109.83, 109.83, 109.83)$, $\sigma = (41.05, 41.05, 41.05)$.
3. **Acoustic Spectral Extraction:**
   * Audio stream demuxed via `ffmpeg` to 16 kHz mono 16-bit PCM WAV.
   * 40-dimensional Mel-Frequency Cepstral Coefficients (MFCC), $N_{fft} = 512$, hop length = 160.
   * 128-dimensional log Mel-scale Spectrogram ($f_{min}=0, f_{max}=8000$ Hz).
   * Feature concatenation produces an input matrix of shape $(168, 300)$ spanning a normalized 3-second temporal slice.

---

## 5. Architectural Framework

### 5.1 Spatial Branch (RGB)
* **Backbone:** ResNeXt-50 ($32 \times 4d$) pretrained on ImageNet-1k via `timm`.
* **Feature Extraction:** Global Average Pooling outputs a 2048-dimensional representation:
  $$\mathbf{h}_{sp} = f_{spatial}(\mathbf{X}_{rgb}) \in \mathbb{R}^{2048}$$
* **Standalone Classifier:** $\text{Dropout}(p=0.5) \to \text{Linear}(2048 \to 1)$.

### 5.2 Frequency Branch (FFT + DWT)
* **Backbone:** Swin Transformer Tiny (`swin_tiny_patch4_window7_224`).
* **Design Justification:** Swin's hierarchical architecture and shifted window-based self-attention ($7 \times 7$ windows) allow modeling both local subband details and cross-window frequency distributions, effectively capturing periodic synthesis artifacts.
* **Feature Extraction:** Global average pooled output yields 768 dimensions:
  $$\mathbf{h}_{freq} = f_{frequency}(\mathbf{X}_{freq}) \in \mathbb{R}^{768}$$
* **Standalone Classifier:** $\text{Dropout}(p=0.5) \to \text{Linear}(768 \to 1)$.

### 5.3 Audio Branch (Speech / Acoustic)
* **Front-End CNN:** 3 convolutional stages (channels 64 $\to$ 128 $\to$ 256, kernel size 3, padding 1) with BatchNorm1d, ReLU, and MaxPool1d($\text{stride}=2$). Captures localized phoneme-level anomalies.
* **Temporal BiLSTM:** 2-layer Bidirectional LSTM ($d_{hidden} = 256$ per direction, 512 total) to model temporal speech dynamics.
* **Temporal Attention Pooling:** Computes normalized attention weights over temporal frames $t \in [1, T']$:
  $$\alpha_t^{time} = \text{Softmax}\left(\mathbf{W}_{att} \mathbf{h}_t^{lstm}\right), \quad \mathbf{h}_{au} = \sum_{t=1}^{T'} \alpha_t^{time} \mathbf{h}_t^{lstm} \in \mathbb{R}^{512}$$
* **Standalone Classifier:** $\text{Dropout}(p=0.3) \to \text{Linear}(512 \to 1)$.

### 5.4 Cross-Attention Multimodal Fusion Network
1. **Latent Projection:** Modalities are mapped into a shared 512-D latent space:
   $$\mathbf{z}_{sp} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{sp} \mathbf{h}_{sp})), \quad \mathbf{z}_{sp} \in \mathbb{R}^{512}$$
   $$\mathbf{z}_{freq} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{freq} \mathbf{h}_{freq})), \quad \mathbf{z}_{freq} \in \mathbb{R}^{512}$$
   $$\mathbf{z}_{au} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{au} \mathbf{h}_{au})), \quad \mathbf{z}_{au} \in \mathbb{R}^{512}$$
2. **Sequence Formulation:** Stacking into a token sequence $\mathbf{Z} = [\mathbf{z}_{sp}, \mathbf{z}_{freq}, \mathbf{z}_{au}] \in \mathbb{R}^{B \times 3 \times 512}$.
3. **Cross-Modality Interaction:** A 1-layer Multi-Head Transformer Encoder ($n_{heads} = 8$, hidden dimension 512, dropout 0.1) performs full cross-attention:
   $$\mathbf{Z}_{fused} = \text{TransformerEncoder}(\mathbf{Z}, \text{padding\_mask}=\mathbf{M})$$
4. **Attention-Based Modality Pooling:**
   $$\mathbf{s} = \mathbf{W}_p \mathbf{Z}_{fused} \in \mathbb{R}^{B \times 3 \times 1}$$
   $$\boldsymbol{\alpha} = \text{Softmax}(\mathbf{s}, \text{dim}=1) = [\alpha_{sp}, \alpha_{freq}, \alpha_{au}]^T$$
   $$\mathbf{c}_{fused} = \sum_{m \in \{sp, freq, au\}} \alpha_m \mathbf{Z}_{fused}^{(m)} \in \mathbb{R}^{B \times 512}$$
5. **Fused Classification Head:**
   $$\hat{y}_{fused} = \mathbf{W}_2 \left( \text{Dropout}_{0.4}(\text{ReLU}(\mathbf{W}_1 \mathbf{c}_{fused})) \right)$$

---

## 6. Training Strategy & Optimization

### 6.1 Multi-Task Auxiliary Objective
To avoid gradient starvation across branches, the total loss combines the fused classification loss with auxiliary unimodal losses:
$$\mathcal{L}_{total} = \mathcal{L}_{BCE}(\hat{y}_{fused}, y) + 0.3 \times \left( \mathbb{I}_{sp} \mathcal{L}_{sp} + \mathbb{I}_{freq} \mathcal{L}_{freq} + \mathbb{I}_{au} \mathcal{L}_{au} \right)$$
where $\mathbb{I}_m$ is an indicator variable indicating whether modality $m$ is valid for the current sample.

### 6.2 Staged Training Schedule
* **Stage 1 (Frozen Backbone Warmup — Epochs 1 to 10):** Unimodal backbones remain frozen. Only the projection heads, Transformer encoder, and classification heads are trained ($\text{LR} = 1.0 \times 10^{-3}$, AdamW).
* **Stage 2 (End-to-End Fine-Tuning — Epochs 11 to 30):** All layers are unfrozen and trained jointly ($\text{LR} = 1.0 \times 10^{-5}$, Cosine Annealing, gradient clipping at 1.0).
* **Modality Dropout:** With $p = 0.10$, a modality is randomly dropped during training to guarantee robustness when streams are missing at test time.

---

## 7. Experimental Results & Benchmarking

### 7.1 Quantitative Performance Comparison

| Model Architecture | Input Modality Streams | ROC-AUC | Accuracy | F1-Score | Precision | Recall | EER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spatial Branch** | RGB Faces ($224 \times 224$) | **0.9676** | 92.95% | 0.9567 | 0.9621 | 0.9512 | 8.94% |
| **Frequency Branch** | FFT + DWT ($3 \times 224 \times 224$) | **0.9703** | **95.10%** | **0.9722** | **0.9703** | **0.9741** | 9.50% |
| **Audio Branch** | MFCC + Mel ($168 \times 300$) | **1.0000** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** |
| **Cross-Attention Fusion** | Spatial + Frequency + Audio | **0.9568** | 94.19% | 0.9669 | 0.9674 | 0.9664 | 11.16% |

### 7.2 Key Analytical Findings
1. **Frequency Branch Outperforms Spatial Branch:** Swin-T on FFT+DWT achieves superior accuracy (95.10% vs. 92.95%) and ROC-AUC (0.9703 vs. 0.9676), validating that frequency decomposition is more resilient than pure RGB color channels.
2. **Audio Sensitivity:** The CNN-BiLSTM architecture detected synthetic speech artifacts in the FakeAVCeleb test split with 100% accuracy and 0.0% EER.
3. **Resilience to Incomplete Modalities:** Thanks to the Boolean modality mask and modality dropout, when audio tracks are unavailable (as in silent FF++ clips), the fusion model dynamically re-weights attention onto spatial and frequency streams without performance collapse.

---

## 8. Probability Calibration & Explainability (XAI)

### 8.1 Platt Scaling Calibration
To convert raw model logits $z$ into true posterior probabilities, Platt scaling is fitted using logistic regression on the held-out validation set:
$$P(\text{Fake} \mid z) = \frac{1}{1 + \exp(-(a \cdot z + b))}$$
Learned parameters ($a, b$) are embedded in the deployment checkpoint, preventing deep network overconfidence.

### 8.2 Modality Attribution & Grad-CAM
* **Modality Attribution Breakdown:** The model natively outputs the attention distribution $\boldsymbol{\alpha} = [\alpha_{sp}, \alpha_{freq}, \alpha_{au}]$. For voice-cloned videos with real video, $\alpha_{au}$ dominates ($> 70\%$). For visual face swaps, $\alpha_{sp}$ and $\alpha_{freq}$ dominate.
* **Grad-CAM Localization:** Gradient-weighted Class Activation Maps on ResNeXt-50 highlight facial boundaries, mouth blending contours, and artifact locations for transparent forensic reports.

---

## 9. Deployment Architecture

* **Inference Engine (`DeepfakePredictor`):** Standalone engine accepting raw MP4 video, performing keyframe face cropping, FFT/DWT decomposition, audio extraction, and calibrated PyTorch inference.
* **Web Demonstration Platform (`app.py`):** Flask web server with interactive UI providing real-time video upload, confidence gauge display, and dynamic modality breakdown charts.

---

## 10. Meta-Prompt for Downstream Paper Writing

```text
You are an expert AI researcher and academic author specializing in Computer Vision, Deep Learning, and Media Forensics.

I have provided you with a comprehensive Master Dossier containing all architectural designs, mathematical formulations, training schedules, datasets, benchmarks, and ablation metrics for my research project on Multimodal Deepfake Detection.

TASK:
Write a complete, highly rigorous, publication-grade research paper formatted according to standard IEEE/ACM conference guidelines. 

INSTRUCTIONS & GUIDELINES:
1. STRUCTURE: Write the full paper including:
   - Title: Tri-Modal Cross-Attention Fusion: Harmonizing Spatial, Spectral, and Acoustic Representations for Robust Deepfake Detection
   - Abstract (refined, high impact, highlighting metrics)
   - Section I: Introduction (background, motivation, key challenges in unimodal deepfake detection, our core contributions)
   - Section II: Related Work (Spatial Artifact Detection, Frequency-Domain Forensics, Audio Deepfake Detection, and Multimodal Fusion Approaches)
   - Section III: Proposed Methodology (mathematical formulations of FFT/DWT generation, 1D-CNN+BiLSTM audio modeling, ResNeXt spatial encoding, Swin-Transformer frequency encoding, Cross-Attention Transformer fusion, and Platt calibration)
   - Section IV: Experimental Setup (dataset descriptions: Celeb-DF v2, FF++ C23, FakeAVCeleb v1.2; preprocessing; training parameters and two-stage schedule)
   - Section V: Experimental Results & Analysis (complete quantitative comparative tables using the exact metrics from the dossier, unimodal vs multimodal analysis, modality masking efficacy)
   - Section VI: Explainability & Interpretability (Attention distribution analysis and Grad-CAM artifact localization)
   - Section VII: Discussion & Limitations (compression degradation, cross-generator generalization, computational complexity)
   - Section VIII: Conclusion & Future Work
2. ACCURACY: Strictly preserve all exact numbers, percentages, dimensions, and model names provided in the dossier (35,089 videos, ResNeXt-50 32x4d, Swin-T, 96.76% spatial AUC, 97.03% frequency AUC, 100.0% audio AUC, 95.68% fusion AUC, db4 wavelets, etc.).
3. MATHEMATICAL RIGOR: Use standard LaTeX equation notation ($...$ and $$...$$).
4. TONE: Objective, academically formal, precise, and authoritative.
```
