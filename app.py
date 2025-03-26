import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import google.generativeai as genai
from google.colab import userdata
from dotenv import load_dotenv
from PIL import Image
import os
load_dotenv()

output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok = True)


def generate_saliency_map(model, img_array, class_index, img_size, original_img, output_dir, uploaded_file):
    # Ensure the image array is a tensor
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)

        # Make predictions
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    # Compute gradients of the target class w.r.t the input image
    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)  # Take absolute value of gradients
    gradients = tf.reduce_max(gradients, axis=-1)  # Take the max along the color channels
    gradients = gradients.numpy().squeeze()

    # Resize gradients to match the original image size
    gradients = cv2.resize(gradients, img_size)

    # Create a circular mask for the brain area
    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

    # Apply mask to gradients
    gradients *= mask

    # Normalize only the brain area
    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min():
        brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
        gradients[mask] = brain_gradients

    # Apply a threshold
    threshold = np.percentile(gradients[mask], 80)
    gradients[gradients < threshold] = 0

    # Apply smoothing
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

    # Create a heatmap overlay with enhanced contrast
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, img_size)

    # Superimpose the heatmap on the original image with increased opacity
    original_img = image.img_to_array(original_img).astype(np.float32)
    superimposed_img = heatmap * 0.7 + original_img * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)

    # Save the saliency map to disk
    img_path = os.path.join(output_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    saliency_map_path = f'saliency_maps/{uploaded_file.name}'
    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return superimposed_img

def load_xception_model(model_path):
    img_shape = (299, 299, 3)
    
    # Define the base model with Xception
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling='max'
    )
    
    # Define the sequential model
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    
    # Load the pre-trained weights
    model.load_weights(model_path)
    
    return model
    

# Streamlit interface
st.title("Brain Tumor Classification")

st.write("Upload an image of a brain MRI scan to classify the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the uploaded file
if uploaded_file is not None:
    select_model = st.radio(
        "Select Model",
        ("Transfer learning - Xception", "Custom CNN")
    )

    # Load the selected model
    if select_model == "Transfer learning - Xception":
        model = load_xception_model("/content/xception_model.weights.h5")
        img_size = (299, 299)
    else:
        model = load_model("/content/cnn_model.h5")
        img_size = (224, 224)

    # Load and preprocess the image
    try:
        img = Image.open(uploaded_file).resize(img_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values

        # Make predictions
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])  # Get the class with the highest probability
        labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
        result = labels[class_index]

        # Display results
        st.write(f"**Predicted Class:** {result}")
        st.write("**Prediction Probabilities:**")
        for label, prob in zip(labels, prediction[0]):
            st.write(f"{label}: {prob:.4f}")
    except Exception as e:
        st.error(f"Error processing the image: {e}")

    saliency_map = generate_saliency_map(model, img_array, class_index, img_size)

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(saliency_map, caption="Saliency Map", use_column_width=True)

    try:
        # Generate saliency map
        saliency_map = generate_saliency_map(
            model=model,
            img_array=img_array,
            class_index=class_index,
            img_size=img_size,
            original_img=img,  # The original PIL Image object
            output_dir=output_dir,  # The directory to save saliency maps
            uploaded_file=uploaded_file  # The uploaded file
        )

        # Display images side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(saliency_map, caption="Saliency Map", use_column_width=True)

    except Exception as e:
        st.error(f"Error generating the saliency map: {e}")
