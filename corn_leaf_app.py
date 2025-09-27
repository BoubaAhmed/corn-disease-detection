import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="Corn Leaf Disease Detection", layout="wide")

# Load model and class names
@st.cache_resource
def load_corn_model():
    try:
        model = load_model("corn_leaf_disease_custom_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_corn_model()

# Class names for corn leaf diseases
class_names = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Sidebar with info
st.sidebar.title("🌽 Corn Leaf Disease Detection")
st.sidebar.markdown(
    """
    **Project: Corn Leaf Disease Classification**
    
    - **Dataset:** [Corn Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
    - **Model:** Custom CNN (384x384 input)
    - **Technology:** TensorFlow, Streamlit
    
    **Disease Classes:**
    - 🍂 **Blight** - Northern Corn Leaf Blight
    - 🟫 **Common Rust** - Common Corn Rust
    - 🔘 **Gray Leaf Spot** - Gray Leaf Spot Disease
    - ✅ **Healthy** - Healthy Corn Leaves
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
st.title("🌽 Corn Leaf Disease Detection")
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:15px;border-radius:8px;">
    <b>Instructions:</b>
    <ul>
      <li>Upload a clear image of a corn leaf</li>
      <li>Supported formats: JPG, JPEG, PNG</li>
      <li>Ensure the leaf is clearly visible and well-lit</li>
      <li>The model will predict the disease type with confidence score</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout: upload + results
col1, col2 = st.columns([2, 1], gap="large")

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

                if model is not None:
                    # Make prediction
                    prediction = model.predict(img_batch, verbose=0)[0]
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100

                    # Display result with appropriate emoji and color
                    if predicted_class == "Healthy":
                        st.success(f"✅ **{predicted_class} Leaf** ({confidence:.2f}% confidence)")
                    else:
                        st.error(f"⚠️ **{predicted_class} Detected** ({confidence:.2f}% confidence)")

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
                    
                    # Recommendations based on prediction
                    st.subheader("💡 Recommendations")
                    if predicted_class == "Healthy":
                        st.success("""
                        **Maintenance Tips:**
                        - Continue regular monitoring
                        - Maintain proper irrigation
                        - Watch for early signs of disease
                        """)
                    elif predicted_class == "Blight":
                        st.warning("""
                        **Action Required:**
                        - Apply fungicides containing azoxystrobin
                        - Remove infected plant debris
                        - Use resistant varieties next season
                        """)
                    elif predicted_class == "Common_Rust":
                        st.warning("""
                        **Action Required:**
                        - Apply fungicides early in season
                        - Ensure proper plant spacing
                        - Avoid overhead irrigation
                        """)
                    elif predicted_class == "Gray_Leaf_Spot":
                        st.warning("""
                        **Action Required:**
                        - Apply appropriate fungicides
                        - Implement crop rotation
                        - Use tillage to bury residue
                        """)
                
                # Show original image in col2
                with col2:
                    st.subheader("🖼️ Image Preview")
                    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
                    
                    # Display disease information
                    st.subheader("🔍 Disease Info")
                    if predicted_class == "Blight":
                        st.info("**Northern Corn Leaf Blight:** Fungal disease causing large lesions on leaves.")
                    elif predicted_class == "Common_Rust":
                        st.info("**Common Rust:** Fungal disease with reddish-brown pustules.")
                    elif predicted_class == "Gray_Leaf_Spot":
                        st.info("**Gray Leaf Spot:** Fungal disease causing rectangular gray spots.")
                    else:
                        st.success("**Healthy Leaf:** No significant disease detected.")

            except Exception as e:
                st.error(f"Error processing image: {e}")
    else:
        st.info("👆 Please upload a corn leaf image to start analysis")

# Sample images expander
with st.expander("🖼️ Sample Images Guide"):
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.markdown("**Blight**")
        st.image("https://media.istockphoto.com/id/1412926967/photo/closeup-corn-leaves-wilting-and-dead-after-wrong-applying-herbicide-in-cornfield.jpg?s=612x612&w=0&k=20&c=1tnV5nsw7yB4HBmtqWS7BTyO3sB1AXgKmWyQ1vGkJhM=", caption="Long gray-green lesions")
    
    with col4:
        st.markdown("**Common Rust**")
        st.image("https://media.istockphoto.com/id/1358387564/photo/stem-rust-also-known-as-cereal-rust-black-rust-red-rust-or-red-dust-is-caused-by-the-fungus.jpg?s=612x612&w=0&k=20&c=sTvrG8-HDu6G1Bc9qvBjHMwtompgSyyiF_vfEHvXM8g=", caption="Reddish-brown pustules")
    
    with col5:
        st.markdown("**Gray Leaf Spot**")
        st.image("https://media.istockphoto.com/id/1934894270/photo/blemish-on-leaf.jpg?s=612x612&w=0&k=20&c=mSDFHD4NnTFQaNp_UjEirIH2c_abbSbZY2tHS9-MxY8=", caption="Rectangular gray spots")
    
    with col6:
        st.markdown("**Healthy**")
        st.image("https://media.istockphoto.com/id/1407403545/photo/corn-leaf-closeup.jpg?s=612x612&w=0&k=20&c=5RgpU9YegU6gthjGowSfJoDBQEYMOnWyWolsJeBRTeI=", caption="Vibrant green leaf")

# Footer
st.markdown("""---""")
st.markdown(
    '<small>🌽 Corn Leaf Disease Detection System • Built with TensorFlow & Streamlit • Agricultural AI Solution</small>',
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
</style>
""", unsafe_allow_html=True)