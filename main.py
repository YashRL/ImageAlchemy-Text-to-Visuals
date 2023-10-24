import streamlit as st
import requests
import cloudinary
import cloudinary.uploader
from PIL import Image
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
from IPython.display import display
import torch as th


# Configure Cloudinary with your credentials
cloudinary.config(
    cloud_name="***********",
    api_key="*************",
    api_secret="***************"
)
#
# # Set up OAuth2 client details
# CLIENT_SECRET_FILE = 'client_secret.json'
# SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']  # Adjust scopes as needed
#
# # Set up Streamlit app
# #st.title("Google Authentication Demo")
#
# # Check if the user is authenticated
# if 'credentials' not in st.session_state:
#     #st.write("WELCOME")
#     flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
#     credentials = flow.run_local_server(port=8501, authorization_prompt_message='')
#
#     # Save credentials to a file for future use (optional)
#     with open('token.json', 'w') as token_file:
#         token_file.write(credentials.to_json())
#
#     st.session_state.credentials = credentials
#     st.success("Authentication successful. You can now use the app.")
#
# # Use authenticated credentials to interact with Google API
# credentials = st.session_state.credentials
# service = build('drive', 'v3', credentials=credentials)
#
# # Fetch user's name from Google API
# try:
#     user_info = service.about().get(fields="user").execute()
#     user_name = user_info["user"]["displayName"]
#     #st.header("Google Profile Information")
#     st.markdown(f"<p style='font-size: 24px;'><strong>Userame: {user_name.upper()}</strong></p>", unsafe_allow_html=True)
# except Exception as e:
#     st.error(f"Error fetching user profile: {str(e)}")
#
# # Your app's functionality goes here
# # # Display Google Drive contents
# # st.header("Google Drive Contents")
# # results = service.files().list(pageSize=10).execute()
# # files = results.get('files', [])
# # if not files:
# #     st.write('No files found in Google Drive.')
# # else:
# #     st.write('Files in Google Drive:')
# #     for file in files:
# #         st.write(f"- {file['name']} ({file['mimeType']})")
#
# # Logout button
# if st.button("Logout"):
#     del st.session_state.credentials
#     os.remove("token_dir/token.json")  # Remove the token file
#



#@title Computer ko aang lagani ho to hi show code click karke ched chad karna

#!pip install git+https://github.com/openai/glide-text2im


from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))

def query_model_with_image(image_description):
    # Sampling parameters
  # image_description = "dog in the field" #@param {type:"string"}
  # image_description = ""
  batch_size = 1 #@param {type:"integer"}
  guidance_scale = 8.0

  # Tune this parameter to control the sharpness of 256x256 images.
  # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
  upsample_temp = 0.997

  ##############################
  # Sample from the base model #
  ##############################

  # Create the text tokens to feed to the model.
  tokens = model.tokenizer.encode(image_description)
  tokens, mask = model.tokenizer.padded_tokens_and_mask(
      tokens, options['text_ctx']
  )

  # Create the classifier-free guidance tokens (empty)
  full_batch_size = batch_size * 2
  uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
      [], options['text_ctx']
  )

  # Pack the tokens together into model kwargs.
  model_kwargs = dict(
      tokens=th.tensor(
          [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
      ),
      mask=th.tensor(
          [mask] * batch_size + [uncond_mask] * batch_size,
          dtype=th.bool,
          device=device,
      ),
  )

  # Create a classifier-free guidance sampling function
  def model_fn(x_t, ts, **kwargs):
      half = x_t[: len(x_t) // 2]
      combined = th.cat([half, half], dim=0)
      model_out = model(combined, ts, **kwargs)
      eps, rest = model_out[:, :3], model_out[:, 3:]
      cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
      half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
      eps = th.cat([half_eps, half_eps], dim=0)
      return th.cat([eps, rest], dim=1)

  # Sample from the base model.
  model.del_cache()
  samples = diffusion.p_sample_loop(
      model_fn,
      (full_batch_size, 3, options["image_size"], options["image_size"]),
      device=device,
      clip_denoised=True,
      progress=True,
      model_kwargs=model_kwargs,
      cond_fn=None,
  )[:batch_size]
  model.del_cache()

  # Show the output
  show_images(samples)


  ##############################
  # Upsample the 64x64 samples #
  ##############################

  tokens = model_up.tokenizer.encode(image_description)
  tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
      tokens, options_up['text_ctx']
  )

  # Create the model conditioning dict.
  model_kwargs = dict(
      # Low-res image to upsample.
      low_res=((samples+1)*127.5).round()/127.5 - 1,

      # Text tokens
      tokens=th.tensor(
          [tokens] * batch_size, device=device
      ),
      mask=th.tensor(
          [mask] * batch_size,
          dtype=th.bool,
          device=device,
      ),
  )

  # Sample from the base model.
  model_up.del_cache()
  up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
  image = diffusion_up.ddim_sample_loop(
      model_up,
      up_shape,
      noise=th.randn(up_shape, device=device) * upsample_temp,
      device=device,
      clip_denoised=True,
      progress=True,
      model_kwargs=model_kwargs,
      cond_fn=None,
  )[:batch_size]
  model_up.del_cache()

  # Show the output
  show_images(image)
  return image


def upload_to_cloudinary(image, prompt_text):
    image_data = io.BytesIO()
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    upload_result = cloudinary.uploader.upload(
        image_data,
        folder="compvis_app",
        public_id=prompt_text
    )
    return upload_result["secure_url"]


def fetch_latest_images_from_cloudinary(num_images=9):
    # Use the Cloudinary Admin API to list resources
    url = f"https://api.cloudinary.com/v1_1/{cloudinary.config().cloud_name}/resources/image"
    params = {
        "max_results": num_images,
        "type": "upload"
    }
    response = requests.get(url, params=params, auth=(cloudinary.config().api_key, cloudinary.config().api_secret))

    if response.status_code == 200:
        images = response.json()["resources"]
    else:
        images = []

    return images

# Streamlit app
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">""", unsafe_allow_html=True)

st.title("Text to Image Generator")

image_description = st.text_input("Enter the image description")

if st.button("Generate Image"):
    processed_image = query_model_with_image(image_description)
    st.image(processed_image, use_column_width=True, output_format="JPEG")  # Use use_column_width=True
    st.session_state.processed_image = processed_image
    st.session_state.image_description = image_description
    st.write("Image generated.")

if st.button("Upload"):
    if 'processed_image' in st.session_state:
        uploaded_url = upload_to_cloudinary(st.session_state.processed_image, st.session_state.image_description)
        st.write("Image uploaded to Cloudinary. Prompt Text:", st.session_state.image_description)
        st.write("Image URL on Cloudinary:", uploaded_url)
    else:
        st.write("Generate an image first before uploading.")

# Fetch and display the latest images from Cloudinary
st.header("Latest Images created")

# Use the 'fetch_latest_images_from_cloudinary' function to get the latest images
latest_images = fetch_latest_images_from_cloudinary()

# Define the number of columns in the grid
num_columns = 3  # You can adjust this number as needed

# Calculate the width for each column
column_width = f"calc(33.33% - {10}px)"  # Adjust the width and margin as needed

# Add CSS styling for the grid and rounded images
st.markdown(
    f"""
    <style>
    .responsive-grid {{
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }}
    .responsive-grid-item {{
        width: {column_width};
        margin: 10px;
        box-sizing: border-box;
        text-align: center;
        position: relative;
    }}
    .image-caption {{
        font-weight: bold;
    }}
    .rounded-image {{
        border-radius: 15px;  # Adjust the radius as needed for more or less roundness
        overflow: hidden;
    }}
    .download-button {{
        background-color: black;  # Set button color to black
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        position: absolute;
        top: 10px;  # Adjust top value for vertical positioning
        right: 10px;  # Adjust right value for horizontal positioning
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Create the responsive grid layout
st.markdown('<div class="responsive-grid">', unsafe_allow_html=True)

for i, image in enumerate(latest_images):
    image_url = image.get('secure_url', '')  # Get the image URL
    public_id = image.get('public_id', '')  # Get the full public_id

    # Extract just the filename (without the folder)
    filename = public_id.split('/')[-1]

    # Add some spacing around the image and its name
    st.markdown(f'<div class="responsive-grid-item">', unsafe_allow_html=True)
    st.markdown(f'<p class="image-caption">{filename}</p>', unsafe_allow_html=True)

    # Add rounded corners to the image using HTML
    st.markdown(f'<img src="{image_url}" class="rounded-image" width="{int(1.25 * 300)}">', unsafe_allow_html=True)

    # Add an arrow icon instead of "Download" button with black color
    download_link = f'<a href="{image_url}" class="download-button" download="{filename}">&#8595;</a>'
    st.markdown(download_link, unsafe_allow_html=True)

    st.write("")  # Add empty spaces for separation
    st.markdown('</div>', unsafe_allow_html=True)

# Close the responsive grid layout
st.markdown('</div>', unsafe_allow_html=True)
