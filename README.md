# 🧠 NeuroSeg AI 🔬

An AI-powered web application for the detection and segmentation of brain tumors from MRI scans, built with a U-Net architecture in TensorFlow and a modern Gradio interface.

**Light Theme:**
<img width="1916" height="912" alt="Screenshot 2025-07-22 205754" src="https://github.com/user-attachments/assets/4fe61bf0-e1fb-4cc9-9de5-58c476be2590" />


**Dark Theme:**
<img width="1911" height="914" alt="Screenshot 2025-07-22 205730" src="https://github.com/user-attachments/assets/a133a177-d8f5-4c7b-8a23-5ff9495ab551" />

---

## 🚀 About The Project

NeuroSeg AI is a deep learning tool designed to provide a fast, accurate, and user-friendly way to analyze medical imaging. It leverages a U-Net model trained on over 3,900 MRI scans to identify and outline tumor regions with high precision. This project showcases a full-cycle machine learning workflow, from training a complex model to deploying it in an interactive and visually appealing web application.

---

## ✨ Features

-   **🧠 Accurate Segmentation:** Utilizes a U-Net model to generate precise, pixel-level masks of predicted tumor regions.
-   **🖥️ Modern UI:** A clean, responsive, and intuitive user interface built with Gradio, featuring a custom blue and white theme.
-   **📊 Rich Analysis:** Provides multiple visual outputs for comprehensive analysis:
    -   **Overlaid Image:** The predicted tumor mask is overlaid in red on the original MRI for easy localization.
    -   **Masked Image:** A clear black-and-white view of the segmented tumor.
-   **📈 Quantitative Results:** Automatically calculates the tumor area as a percentage of the total brain area and provides a confidence score.
-   **⬇️ Downloadable Reports:** Users can download a text file summary of the analysis for their records.

---

## 🛠️ Tech Stack

This project was built using the following technologies:

-   **TensorFlow & Keras:** For building and training the U-Net deep learning model.
-   **Gradio:** For creating and deploying the interactive web interface.
-   **OpenCV & NumPy:** For high-performance image processing and data manipulation.

---

## 🏁 Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites

-   Python 3.9+
-   An environment manager like `venv`

### Quick Start Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aaryan-paliwal/NeuroSeg-AI.git
    cd NeuroSeg-AI
    ```

2.  **Download the Trained Model:**
    The trained model (`best_model.keras`) is too large for this repository. Please download it from [best_model.keras](https://drive.google.com/drive/folders/1OSbjsM4S-TVhHs2ku9J0N-NJRfnSXEuI?usp=drive_link) and place it in the main project directory.

3.  **Set up the Python environment:**
    ```bash
    # Create a virtual environment
    python -m venv app_env

    # Activate it (Windows)
    .\app_env\Scripts\Activate.ps1

    # Install the required packages
    pip install -r requirements.txt
    ```

4.  **Launch the application:**
    ```bash
    python app.py
    ```
    Open your browser and navigate to the local URL provided in the terminal (e.g., `http://127.0.0.1:7860`).

---

## 📧 Contact & Support
For questions or to support:
- Author: Aaryan Paliwal
- Email: aapaliwal.work@gmail.com
- LinkedIn Profile: https://www.linkedin.com/in/aaryan-paliwal/

