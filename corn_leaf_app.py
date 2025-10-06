import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Corn Leaf Disease Detection", layout="wide")

# Load model and class names
@st.cache_resource
def load_corn_model():
    try:
        model = load_model("corn_leaf_disease_model_XAI.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_corn_model()

# Class names for corn leaf diseases
class_names = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# ========== XAI FUNCTIONS ==========

def get_last_conv_layer(model):
    """Find the last convolutional layer"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    
    # If no Conv2D found, search in nested layers
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    
    raise ValueError("No convolutional layer found!")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(conv_outputs, (list, tuple)):
                conv_outputs = conv_outputs[0]
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy(), int(pred_index)

    except Exception as e:
        st.error(f"❌ Error in Grad-CAM: {e}")
        return None, None

def generate_xai_visualization(model, img_array, original_img, pred_class, confidence, class_names):
    """Generate XAI visualization with Grad-CAM"""
    try:
        # Find the last convolutional layer
        last_conv_layer_name = get_last_conv_layer(model)
        
        # Generate heatmap
        heatmap, _ = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class)
        
        if heatmap is None:
            st.error("Could not generate heatmap")
            return None
        
        # Convert PIL Image to numpy array for OpenCV
        original_img_np = np.array(original_img)
        
        # Resize heatmap and create overlay
        heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Convert original image to BGR for overlay
        original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
        superimposed_img = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored, 0.4, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM Overlay
        axes[1].imshow(superimposed_img_rgb)
        axes[1].set_title('Grad-CAM Overlay\n(Areas the model focused on)', fontweight='bold')
        axes[1].axis('off')
        
        # Heatmap only
        axes[2].imshow(heatmap, cmap='jet')
        axes[2].set_title('Heatmap Only', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error generating XAI visualization: {e}")
        return None

# ========== STREAMLIT UI ==========

# Sidebar with info
st.sidebar.title("🌽 Corn Leaf Disease Detection")
st.sidebar.markdown(
    """
    **Project: Corn Leaf Disease Classification with XAI**
    
    - **Dataset:** [Corn Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
    - **Model:** Custom CNN (384x384 input)
    - **Technology:** TensorFlow, Streamlit, Grad-CAM
    
    **Disease Classes:**
    - 🍂 **Blight** - Northern Corn Leaf Blight
    - 🟫 **Common Rust** - Common Corn Rust
    - 🔘 **Gray Leaf Spot** - Gray Leaf Spot Disease
    - ✅ **Healthy** - Healthy Corn Leaves
    
    **XAI Feature:**
    - 🔍 **Grad-CAM** visualization shows which parts of the image the model used for prediction
    """
)
st.sidebar.info(
    "This application helps farmers and agricultural experts quickly identify corn leaf diseases for early treatment."
)

# Disease information expander
with st.sidebar.expander("📋 Disease Information"):
    st.markdown("""
    **Northern Corn Leaf Blight:**
    - Symptoms: Long, elliptical gray-green lesions
    - Treatment: Fungicides, resistant varieties
    
    **Common Corn Rust:**
    - Symptoms: Small, circular reddish-brown pustules
    - Treatment: Fungicide applications
    
    **Gray Leaf Spot:**
    - Symptoms: Rectangular, tan to gray spots
    - Treatment: Crop rotation, fungicides
    """)

# Main Title & Instructions
st.title("🌽 Corn Leaf Disease Detection with XAI")
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:15px;border-radius:8px;">
    <b>Instructions:</b>
    <ul>
      <li>Upload a clear image of a corn leaf</li>
      <li>Supported formats: JPG, JPEG, PNG</li>
      <li>Ensure the leaf is clearly visible and well-lit</li>
      <li>The model will predict the disease type with confidence score</li>
      <li><b>NEW:</b> XAI visualization shows which areas influenced the prediction</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout: upload + results
col1, col2 = st.columns([2, 1], gap="large")

# Initialize variables
predicted_class_name = None
prediction = None
img = None

with col1:
    st.subheader("📤 Upload Corn Leaf Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Analyzing leaf image..."):
            try:
                # Load and preprocess image
                img = Image.open(uploaded_file).convert("RGB")
                
                # Display image info
                st.info(f"Image size: {img.size} | Format: {img.format}")
                
                # Resize to model input size (384x384)
                img_resized = img.resize((384, 384))
                img_array = image.img_to_array(img_resized) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Predict
                prediction = model.predict(img_batch, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class] * 100

                if predicted_class is not None:
                    predicted_class_name = class_names[predicted_class]
                    
                    # Display result with appropriate emoji and color
                    if predicted_class_name == "Healthy":
                        st.success(f"✅ **{predicted_class_name} Leaf** ({confidence:.2f}% confidence)")
                    else:
                        st.error(f"⚠️ **{predicted_class_name} Detected** ({confidence:.2f}% confidence)")

                    # Progress bars for probabilities
                    st.subheader("📊 Detailed Probabilities")
                    
                    for i, (class_name, prob) in enumerate(zip(class_names, prediction)):
                        percentage = prob * 100
                        col_a, col_b = st.columns([1, 3])
                        
                        with col_a:
                            st.write(f"**{class_name}:**")
                        with col_b:
                            st.progress(float(prob))
                            st.write(f"{percentage:.2f}%")
                    
                    # XAI Visualization
                    st.subheader("🔍 Model Explainability (XAI)")
                    st.info("The Grad-CAM visualization shows which parts of the image the model focused on to make its prediction.")
                    
                    xai_fig = generate_xai_visualization(
                        model, img_batch, img, predicted_class, confidence, class_names
                    )
                    
                    if xai_fig:
                        st.pyplot(xai_fig)
                        st.caption("""
                        **How to interpret the XAI visualization:**
                        - **Red/Orange areas:** Regions that strongly influenced the model's prediction
                        - **Blue areas:** Regions that had less influence on the decision
                        - The model should focus on diseased areas of the leaf for accurate diagnosis
                        """)
                    
                    # Recommendations based on prediction
                    st.subheader("💡 Recommendations")
                    if predicted_class_name == "Healthy":
                        st.success("""
                        **Maintenance Tips:**
                        - Continue regular monitoring
                        - Maintain proper irrigation
                        - Watch for early signs of disease
                        """)
                    elif predicted_class_name == "Blight":
                        st.warning("""
                        **Action Required:**
                        - Apply fungicides containing azoxystrobin
                        - Remove infected plant debris
                        - Use resistant varieties next season
                        """)
                    elif predicted_class_name == "Common_Rust":
                        st.warning("""
                        **Action Required:**
                        - Apply fungicides early in season
                        - Ensure proper plant spacing
                        - Avoid overhead irrigation
                        """)
                    elif predicted_class_name == "Gray_Leaf_Spot":
                        st.warning("""
                        **Action Required:**
                        - Apply appropriate fungicides
                        - Implement crop rotation
                        - Use tillage to bury residue
                        """)
                else:
                    st.error("❌ Could not generate prediction")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.info("Please try with a different image or check the file format.")

# Show original image and additional info in col2
with col2:
    if uploaded_file is not None and img is not None:
        st.subheader("🖼️ Image Preview")
        st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
        
        # Display disease information
        if predicted_class_name is not None:
            st.subheader("🔍 Disease Info")
            if predicted_class_name == "Blight":
                st.info("""
                **Northern Corn Leaf Blight:**
                - Fungal disease causing large lesions
                - Spread by wind and rain
                - Affects yield significantly
                """)
            elif predicted_class_name == "Common_Rust":
                st.info("""
                **Common Rust:**
                - Fungal disease with reddish-brown pustules
                - Thrives in cool, humid conditions
                - Can reduce photosynthesis
                """)
            elif predicted_class_name == "Gray_Leaf_Spot":
                st.info("""
                **Gray Leaf Spot:**
                - Fungal disease causing rectangular spots
                - Common in continuous corn fields
                - Managed with fungicides
                """)
            else:
                st.success("""
                **Healthy Leaf:**
                - No significant disease detected
                - Continue good agricultural practices
                - Monitor regularly
                """)
        
        # Model confidence metrics
        if prediction is not None:
            st.subheader("📈 Confidence Metrics")
            data = {
                'Disease': class_names,
                'Confidence': [f"{p*100:.1f}%" for p in prediction]
            }
            st.table(data)
    else:
        st.info("👆 Upload an image to see preview and details")

# Sample images expander
with st.expander("🖼️ Sample Images Guide"):
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.markdown("**Blight**")
        st.image("https://media.istockphoto.com/id/1412926967/photo/closeup-corn-leaves-wilting-and-dead-after-wrong-applying-herbicide-in-cornfield.jpg?s=612x612&w=0&k=20&c=1tnV5nsw7yB4HBmtqWS7BTyO3sB1AXgKmWyQ1vGkJhM=", 
                 caption="Long gray-green lesions")
    
    with col4:
        st.markdown("**Common Rust**")
        st.image("https://media.istockphoto.com/id/1358387564/photo/stem-rust-also-known-as-cereal-rust-black-rust-red-rust-or-red-dust-is-caused-by-the-fungus.jpg?s=612x612&w=0&k=20&c=sTvrG8-HDu6G1Bc9qvBjHMwtompgSyyiF_vfEHvXM8g=", 
                 caption="Reddish-brown pustules")
    
    with col5:
        st.markdown("**Gray Leaf Spot**")
        st.image("https://media.istockphoto.com/id/1934894270/photo/blemish-on-leaf.jpg?s=612x612&w=0&k=20&c=mSDFHD4NnTFQaNp_UjEirIH2c_abbSbZY2tHS9-MxY8=", 
                 caption="Rectangular gray spots")
    
    with col6:
        st.markdown("**Healthy**")
        st.image("https://media.istockphoto.com/id/1407403545/photo/corn-leaf-closeup.jpg?s=612x612&w=0&k=20&c=5RgpU9YegU6gthjGowSfJoDBQEYMOnWyWolsJeBRTeI=", 
                 caption="Vibrant green leaf")

# Footer
st.markdown("""---""")
st.markdown(
    '<small>🌽 Corn Leaf Disease Detection System • Built with TensorFlow & Streamlit • Agricultural AI Solution • Now with Model Explainability (XAI)</small>',
    unsafe_allow_html=True
)

# Add some CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
    }
    .prediction-result {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
</style>
""", unsafe_allow_html=True)