import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import io

# Set up device and models
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)
    return device, pipe, clip_model, tokenizer

device, pipe, clip_model, tokenizer = load_models()

def generate_image_from_flavor(flavor_description):
    prompt = f"A vibrant, artistic representation of {flavor_description}. Digital art, colorful, abstract, food illustration."
    inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = clip_model(**inputs.to(device)).last_hidden_state
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    return image

# Streamlit app
st.title("Flavor-Inspired Image Generator")

# Input for flavor description
flavor_description = st.text_input("Enter a flavor description:", "A refreshing summer drink with notes of citrus and mint")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image_from_flavor(flavor_description)
        st.image(image, caption=flavor_description, use_column_width=True)
        
        # Option to download the image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        btn = st.download_button(
            label="Download Image",
            data=buf.getvalue(),
            file_name=f"{flavor_description.replace(' ', '_')}.png",
            mime="image/png"
        )

# Predefined flavor descriptions
st.sidebar.header("Try these flavors:")
predefined_flavors = [
    "A refreshing summer drink with notes of citrus and mint",
    "A decadent dessert combining the flavors of dark chocolate and raspberry",
    "An aromatic coffee blend featuring hints of caramel and cinnamon",
    "A savory appetizer that balances the tastes of aged cheese and truffle",
    "A unique ice cream flavor inspired by lavender and honey"
]

for flavor in predefined_flavors:
    if st.sidebar.button(flavor):
        st.text_input("Enter a flavor description:", value=flavor, key="flavor_input")
        st.experimental_rerun()