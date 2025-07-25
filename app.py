# --- 0. Import Necessary Libraries ---
# Gradio for the web interface, TensorFlow for the model, NumPy/OpenCV for image processing.
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

# --- 1. Configuration and Setup ---
# Define constants for easy configuration.
MODEL_PATH = 'best_model.keras'
IMAGE_SIZE = 256

# --- 2. Load the Trained Model with Error Handling ---
# We use a try-except block to gracefully handle errors if the model file is missing.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"---! ERROR LOADING MODEL !---")
    print(f"Error details: {e}")
    model = None

# --- 3. Custom CSS for UI Styling ---
# This block contains all the styling rules for our app's appearance.
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

/* --- NEW: Light blue gradient background to force light theme --- */
body, .gradio-container {
    background: linear-gradient(135deg, #e3f0ff 0%, #b3d1f7 100%) !important;
    font-family: 'Montserrat', 'Poppins', Arial, sans-serif;
    color: #0a2540;
}

#main-title {
    text-align: center; color: #0a2540; font-size: 2.7em; font-weight: 700;
    letter-spacing: 1px; margin-bottom: 0.2em; text-shadow: 0 2px 8px rgba(179, 209, 247, 0.5);
}
#subtitle {
    text-align: center; color: #2563eb; font-size: 1.2em;
    margin-bottom: 28px; font-weight: 500;
}

/* --- NEW: Prominent white boxes with stronger borders and shadows --- */
.gradio-group, .gradio-tabs, .gr-panel {
    border-radius: 16px !important;
    background: white !important; /* Pure white boxes */
    box-shadow: 0 8px 32px 0 rgba(30, 64, 175, 0.15) !important; /* More prominent shadow */
    border: 2px solid #60a5fa !important; /* Stronger blue border */
}

.gr-button {
    background: linear-gradient(90deg, #2563eb 0%, #60a5fa 100%) !important;
    color: #fff !important; border-radius: 10px !important; border: none !important;
    font-weight: 600 !important; font-size: 1.1em !important; padding: 0.7em 2.2em !important;
    box-shadow: 0 2px 8px 0 rgba(179, 209, 247, 0.8) !important; transition: background 0.2s, box-shadow 0.2s;
}
.gr-button:hover {
    background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%) !important;
    box-shadow: 0 4px 16px 0 #60a5fa !important;
}
.gr-label {
    font-weight: 700 !important; color: #2563eb !important;
    font-family: 'Montserrat', 'Poppins', Arial, sans-serif; letter-spacing: 0.5px;
}
.gr-image {
    border-radius: 12px !important; border: 2px solid #b3d1f7 !important;
    background: #fafdff !important; box-shadow: 0 2px 12px 0 rgba(179, 209, 247, 0.5) !important;
}
"""

# --- 4. Core Prediction Function (No changes) ---
def predict(input_image):
    # Check if the model is loaded and an image was provided.
    if model is None:
        raise gr.Error("Model is not loaded. Check terminal for errors.")
    if input_image is None:
        raise gr.Error("Please upload an image before submitting.")
    
    # Preprocess the image to the format the model expects (256x256, normalized).
    image_resized = cv2.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE))
    image_normalized = image_resized / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    # Get the raw prediction from the model.
    predicted_mask_raw = model.predict(image_batch)[0]

    # Convert the prediction probabilities to a binary mask (0 or 1).
    binary_mask = (predicted_mask_raw > 0.5).astype(np.uint8)

    # --- Create Visual Outputs ---
    colored_mask = np.zeros_like(image_resized)
    colored_mask[binary_mask.squeeze() == 1] = [220, 20, 60]
    overlay_image = cv2.addWeighted(image_resized, 0.7, colored_mask, 0.3, 0)
    masked_image = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
    
    # --- Create Textual Summary ---
    tumor_pixels = np.sum(binary_mask)
    if tumor_pixels > 0:
        prediction_text = "Tumor Detected"
        probability_text = f"{np.mean(predicted_mask_raw[binary_mask == 1]) * 100:.2f}%"
        tumor_area_text = f"{(tumor_pixels / binary_mask.size) * 100:.2f}%"
    else:
        prediction_text = "No Tumor Detected"
        probability_text = "N/A"
        tumor_area_text = "0.00%"
        
    # --- Create Downloadable Report ---    
    report_content = (f"Analysis Report\n{'='*17}\nPrediction: {prediction_text}\nConfidence: {probability_text}\nTumor Area: {tumor_area_text}\n")
    with open("report.txt", "w") as f:
        f.write(report_content)

    # Return all the results in a dictionary to update the UI.
    return {
        results_group: gr.update(visible=True),
        output_overlay: overlay_image,
        output_mask: masked_image,
        output_original: input_image,
        prediction_out: prediction_text,
        probability_out: probability_text,
        area_out: tumor_area_text,
        download_report: "report.txt"
    }

# --- 5. Function to Reset/Clear the UI  ---
def clear_interface():
    return {
        results_group: gr.update(visible=False),
        input_image: None
    }

# --- 6. Gradio UI Definition ---
# gr.Blocks provides full control over the layout of the web app.
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    # Header section with a title and subtitle.
    gr.Markdown("<h1 id='main-title'>NeuroSeg AI</h1>", elem_id="main-title")
    gr.Markdown("<p id='subtitle'>AI-Powered Brain Tumor Detection & Segmentation</p>", elem_id="subtitle")
    
    # Input section with the image uploader.
    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload Brain MRI")
    # Buttons for submitting and clearing.
    with gr.Row():
        cancel_button = gr.Button("Clear")
        submit_button = gr.Button("Submit", variant="primary")

    # Results section, which starts hidden.
    with gr.Group(visible=False) as results_group:
        
        # A row to display the three output images.
        with gr.Row(variant='panel'):
            output_overlay = gr.Image(label="Overlaid Image")
            output_mask = gr.Image(label="Masked Image")
            output_original = gr.Image(label="Original Image")
        
        # A group for the textual summary.
        with gr.Group(elem_id="result-summary"):
            gr.Markdown("## Result Summary")
            with gr.Row():
                prediction_out = gr.Textbox(label="Prediction")
                probability_out = gr.Textbox(label="Confidence")
                area_out = gr.Textbox(label="Tumor Area")

        # A component for downloading the report file.
        download_report = gr.File(label="Download Full Report")


    # --- Define Component Interactions ---
    # When the submit button is clicked, run the prediction function.
    submit_button.click(
        fn=predict, 
        inputs=input_image, 
        outputs=[results_group, output_overlay, output_mask, output_original, prediction_out, probability_out, area_out, download_report]
    )
    # When the cancel button is clicked, run the clear function.
    cancel_button.click(
        fn=clear_interface, 
        inputs=None, 
        outputs=[results_group, input_image]
    )

# --- 7. Launch the Web App ---
# The if __name__ == "__main__": block ensures this code only runs when the script is executed directly.
if __name__ == "__main__":
    if model: # Only launch if the model was loaded successfully.
        # Get the port from the environment variable Render provides.
        port = int(os.environ.get('PORT', 7860))

        # Launch the app to be accessible on Render's network.
        demo.launch(server_name="0.0.0.0", server_port=port)
    else:
        print("\nApplication launch failed because the model could not be loaded.")
