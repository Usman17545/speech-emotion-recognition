# 🎙️ Speech Emotion Recognition (Urdu)

This project implements a **Speech Emotion Recognition (SER)** system using **deep learning (LSTMs)** on **Urdu audio data**. It classifies speech into four emotions:

- 😀 Happy (خوش)  
- 😢 Sad (اداس)  
- 😡 Angry (غصہ)  
- 😐 Neutral (غیر جانبدار)  

The model takes audio files as input, extracts features, and predicts the emotion expressed in the speech.

---

## 📊 Dataset
- Dataset: **Urdu Speech Emotion Dataset**  
- Format: `.wav` audio files  
- Each emotion is stored in a separate folder:
  - `happy` (خوش)  
  - `sad` (اداس)  
  - `angry` (غصہ)  
  - `neutral` (غیر جانبدار)  

- Total audio samples: ~XXXX (replace with your dataset size)  
- Sample rate: 16 kHz  
- Duration per sample: variable, padded/truncated to 2–3 seconds for uniform input

---

## ⚙️ Feature Extraction
- Features extracted using **MFCCs (Mel-Frequency Cepstral Coefficients)**  
- Number of MFCCs: 40  
- Each audio file is converted into a feature matrix of shape `(40, 200)` (padded/truncated to 200 frames)  
- Feature extraction steps:
  1. Load audio using **librosa** at 16 kHz
  2. Compute 40 MFCCs
  3. Pad or truncate each sample to ensure a fixed shape
- Example code:

## 🧠 Model Architecture

The LSTM model captures **temporal patterns** in speech to classify emotions. We use **Dropout** and **Batch Normalization** to improve generalization and stabilize training.

**Architecture details:**

- **LSTM layer (128 units, return_sequences=True)**  
  - Processes sequences of MFCC frames and outputs a sequence of hidden states  
  - Helps capture temporal dependencies in the audio  

- **Dropout (0.3)**  
  - Randomly drops 30% of units during training to prevent overfitting  

- **Batch Normalization**  
  - Normalizes the outputs from LSTM to stabilize and accelerate training  

- **Second LSTM layer (64 units)**  
  - Further processes the sequence of features  

- **Dropout + Batch Normalization**  
  - Adds regularization and normalization again  

- **Dense layer (64 units, ReLU)**  
  - Fully connected layer to combine features before classification  

- **Dropout (0.3)**  
  - Another regularization layer  

- **Output layer (4 units, Softmax)**  
  - Predicts probabilities for each of the four emotions
  
**Model Performance:**  
- Achieved an **accuracy of ~85%** on the test set.
