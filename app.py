import streamlit as st
import numpy as np
import torch
import torch.nn as nn

# 1. Define the exact same architecture used during training
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=31141, output_dim=6144):
        super(BaselineMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.network(x)

# 2. Main Streamlit Dashboard UI Layout
st.title("🧠 fMRI to 3D Object Reconstruction")
st.write("Upload a test fMRI data file to visualize the reconstructed 3D shape.")

# File uploader widget
uploaded_file = st.file_uploader("Upload fMRI Scan (.npz File)", type=["npz"])

if uploaded_file is not None:
    st.info("Processing fMRI scan through the PyTorch model...")
    
    # Load data safely from buffer
    data = np.load(uploaded_file)
    if 'X' in data.files:
        # Preprocess features
        fMRI_input = np.mean(data['X'], axis=0 if len(data['X'].shape) == 2 else 1)
        fMRI_tensor = torch.tensor(fMRI_input, dtype=torch.float32).unsqueeze(0)
        
        # Initialize model architecture setup
        model = BaselineMLP(input_dim=31141, output_dim=6144)
        model.eval()
        
        with torch.no_grad():
            pred_flat = model(fMRI_tensor).numpy().flatten()
            
        points = pred_flat.reshape(2048, 3)
        
        st.success("3D Point Cloud Generated Successfully!")
        
        # Display the shapes of individual arrays to verify metrics
        st.write(f"Total Predicted Vertices: {points.shape} points")
        st.dataframe(points[:10], column_config={"0": "X", "1": "Y", "2": "Z"})
        
        # Quick fallback option to download the final PLY file directly
        ply_header = f"ply\nformat ascii 1.0\nelement vertex 2048\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        ply_body = "".join([f"{p:.6f} {p:.6f} {p:.6f}\n" for p in points])
        st.download_button("📥 Download Reconstructed .PLY Mesh File", data=ply_header + ply_body, file_name="output.ply")
    else:
        st.error("Invalid File Format: Field 'X' not found in the uploaded archive.")
