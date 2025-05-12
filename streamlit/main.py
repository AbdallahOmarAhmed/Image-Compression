import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import hashlib
from ImageCompression import ImageCompression
from torchvision import transforms

# XOR-based cipher
def xor_cipher(data: bytes, password: str) -> bytes:
    key = hashlib.sha256(password.encode('utf-8')).digest()
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))

# Cache model loading
def load_model():
    model = ImageCompression.load_from_checkpoint("model.ckpt")
    model.eval()
    return model
load_model = st.cache_resource(load_model)

# Initialize transforms
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
model = load_model()

st.title("AI Based Image Compression")

# Single uploader for both actions
uploaded = st.file_uploader("Upload file to compress or decompress", type=None)
# Password input
encryption_password = st.text_input("Enter password", type="password")

# Buttons layout
col1, col2 = st.columns(2)

# Compress flow
with col2:
    if uploaded and encryption_password and st.button("Compress"):
        try:
            # Open image
            orig_img = Image.open(uploaded).convert("RGB")
            # Prepare tensor for encoder
            img_tensor = to_tensor(orig_img).unsqueeze(0)
            x_norm = (img_tensor - 0.5) / 0.5
            # Encode
            with torch.no_grad():
                latent = model.encoder(x_norm)
            comp_img = to_pil(latent.squeeze(0))
            # Serialize to JPEG
            buf = BytesIO()
            comp_img.save(buf, format="JPEG", quality=85)
            comp_bytes = buf.getvalue()
            # Encrypt
            encrypted = xor_cipher(comp_bytes, encryption_password)
            # Stats
            orig_bytes = uploaded.getvalue()
            st.write(f"Original size: {len(orig_bytes):,} bytes")
            st.write(f"Encrypted compressed size: {len(encrypted):,} bytes")
            st.write(f"Reduction: {len(orig_bytes) - len(encrypted):,} bytes ({(len(orig_bytes)-len(encrypted))/len(orig_bytes):.2%})")
            # Download
            st.download_button(
                "Download Encrypted Compressed File",
                data=encrypted,
                file_name="compressed_encrypted.bin",
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Compression failed: {e}")

# Decompress flow
with col1:
    if uploaded and encryption_password and st.button("Decompress"):
        try:
            # Read uploaded bytes
            encrypted_bytes = uploaded.getvalue()
            # Decrypt
            decrypted = xor_cipher(encrypted_bytes, encryption_password)
            # Load decrypted bytes as image
            buf = BytesIO(decrypted)
            comp_img_in = Image.open(buf).convert("RGB")
            # Preprocess for decoder (orig code: x = totensor(orig) -0.5)
            comp_tensor = to_tensor(comp_img_in).unsqueeze(0)
            x = comp_tensor - 0.5
            # Decode
            with torch.no_grad():
                out = model.decoder(x) + 1
            decomp_img = to_pil(out.squeeze(0) / 2)
            # Display result
            st.subheader("Decompressed Output")
            st.image(decomp_img, use_container_width=True)
            # Download
            buf2 = BytesIO()
            decomp_img.save(buf2, format="JPEG", quality=85)
            decomp_bytes = buf2.getvalue()
            st.download_button(
                "Download Decompressed Image",
                data=decomp_bytes,
                file_name="decompressed.jpg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"Decompression failed: {e}")

# Warnings
if uploaded and not encryption_password:
    st.warning("Please enter a password before proceeding.")
