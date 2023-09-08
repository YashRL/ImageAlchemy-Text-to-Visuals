import streamlit as st
import requests
import cloudinary
import cloudinary.uploader
from PIL import Image
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

# Configure Cloudinary with your credentials
cloudinary.config(
    cloud_name="dvuowbmrz",
    api_key="177664162661619",
    api_secret="qVMYel17N_C5QUUUuBIuatB5tq0"
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


# Set up Hugging Face API endpoint
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers = {"Authorization": "Bearer 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"}


def query_model_with_image(image_description):
    payload = {
        "inputs": image_description
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    image_bytes = response.content

    image = Image.open(io.BytesIO(image_bytes))
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