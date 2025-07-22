# 🧠 NeuroSeg AI 🔬

An AI-powered web application for the detection and segmentation of brain tumors from MRI scans, built with a U-Net architecture in TensorFlow and a modern Gradio interface.

![App Screenshot](https://i.imgur.com/YOUR_SCREENSHOT_URL.png) 
*Note: Please take a screenshot of your beautiful final app, upload it to a site like [Imgur](https://imgur.com/upload), and replace the link above.*

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
    git clone [https://github.com/aaryan-paliwal/NeuroSeg-AI.git](https://github.com/aaryan-paliwal/NeuroSeg-AI.git)
    cd NeuroSeg-AI
    ```

2.  **Download the Trained Model:**
    The trained model (`best_model.keras`) is too large for this repository. Please download it from [LINK_TO_YOUR_MODEL] and place it in the main project directory.
    *(**Note**: You can upload your `.keras` file to a service like Google Drive or Dropbox and share the link here.)*

3.  **Set up the Python environment:**
    ```bash
    # Create a virtual environment
    python -m venv app_env

    # Activate it (Windows)
    .\app_env\Scripts\activate

    # Install the required packages
    pip install -r requirements.txt
    ```

4.  **Launch the application:**
    ```bash
    python app.py
    ```
    Open your browser and navigate to the local URL provided in the terminal (e.g., `http://127.0.0.1:7860`).

---

## 📧 Contact

Aaryan Paliwal - aapaliwal.work@gmail.com
Project Link: [https://github.com/aaryan-paliwal/NeuroSeg-AI](https://github.com/aaryan-paliwal/NeuroSeg-AI)