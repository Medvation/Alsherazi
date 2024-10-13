import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import traceback
import gc

st.set_page_config(layout="wide", page_title="AI-JAM by Alsherazi", page_icon="🎨")

# Custom CSS for the updated look
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
    }
    .big-font {
        font-size: 60px !important;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px 0 0 0;
        margin-bottom: 0;
    }
    .subheader {
        font-size: 30px;
        font-style: italic;
        color: white;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ecf0f1;
        color: black !important;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-JAM</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator by Alsherazi Club</p>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)
    return device, pipe

device, pipe = load_models()

def generate_image_from_flavor(flavor_description, progress_bar, status_text):
    prompt = f"A vibrant, artistic representation of {flavor_description}. Digital art, colorful, abstract, food illustration, family-friendly, non-offensive."
    
    num_inference_steps = 20
    
    def callback(step: int, timestep: int, latents: torch.FloatTensor):
        progress = min((step + 1) / num_inference_steps, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Generating image... {progress:.0%}")
        return
    
    try:
        with torch.no_grad():
            output = pipe(
                prompt, 
                num_inference_steps=num_inference_steps, 
                callback=callback, 
                callback_steps=1,
                guidance_scale=7.5,
                negative_prompt="nsfw, offensive content, low quality, blurry"
            )
        image = output.images[0]
        return image
    except Exception as e:
        st.error(f"Error in image generation: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Flavor menu
st.sidebar.markdown("### 🍽️ Flavor Palette")
predefined_flavors = [
    "A refreshing summer drink with notes of citrus and mint",
    "A decadent dessert combining dark chocolate and raspberry",
    "An aromatic coffee blend with hints of caramel and cinnamon",
    "A savory appetizer balancing aged cheese and truffle",
    "A unique ice cream inspired by lavender and honey"
]

flavor_menu = st.sidebar.empty()
selected_flavor = flavor_menu.radio("Select a flavor inspiration or create your own:", 
                                    ["Create your own"] + predefined_flavors, 
                                    index=0,
                                    format_func=lambda x: x if x != "Create your own" else "✨ Create your own flavor")

# Main area
if selected_flavor == "Create your own":
    flavor_description = st.text_input("Enter your flavor description:", "", key="flavor_input")
else:
    flavor_description = st.text_input("Enter your flavor description:", selected_flavor, key="flavor_input")

if st.button("🎨 Generate Artistic Impression"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Preparing to generate image...")
        image = generate_image_from_flavor(flavor_description, progress_bar, status_text)
        
        if image is not None:
            status_text.text("Image generation complete!")
            st.image(image, caption=f"Artistic impression of: {flavor_description}", use_column_width=True)
            
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Your Artwork",
                data=buf.getvalue(),
                file_name=f"{flavor_description.replace(' ', '_')}_art.png",
                mime="image/png"
            )
        else:
            st.error("Failed to generate a valid image. Please try again with a different description.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please try again or contact support if the issue persists.")
        st.error("Detailed error information:")
        st.code(traceback.format_exc())
    finally:
        status_text.text("Process completed.")
        progress_bar.empty()  # Clear the progress bar
        gc.collect()
        torch.cuda.empty_cache()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-JAM")
st.sidebar.markdown("AI-JAM is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence. Our goal is to create a unique sensory experience by translating flavors into visual art.")