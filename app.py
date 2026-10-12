import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import os

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

# 2. Prediction function for the web interface
def predict_3d_object(file_obj):
    if file_obj is None:
        return None

    # Load the uploaded .npz fMRI file safely
    data = np.load(file_obj.name)
    if 'X' not in data.files:
        return None

    # Preprocess: Take the mean over time points and convert to tensor
    fmli_input = np.mean(data['X'], axis=0 if len(data['X'].shape) == 2 else 1)
    fmli_tensor = torch.tensor(fmli_input, dtype=torch.float32).unsqueeze(0)

    # Run the model (Using the instantiated model object from the main environment)
    model.eval()
    with torch.no_grad():
        pred_flat = model(fmli_tensor).numpy().flatten()

    # Reshape to 3D points (2048, 3)
    points = pred_flat.reshape(2048, 3)

    # Save as a temporary PLY file to display in the 3D viewer
    output_path = "predicted_output.ply"
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 2048\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for pt in points:
            f.write(f"{float(pt[0]):.6f} {float(pt[1]):.6f} {float(pt[2]):.6f}\n")

    return output_path

# 3. Create the Gradio Interface layout
demo = gr.Interface(
    fn=predict_3d_object,
    inputs=gr.File(file_types=[".npz"], label="Upload fMRI Scan (.npz File)"),
    outputs=gr.Model3D(label="Predicted 3D Object Reconstruction"),
    title="fMRI to 3D Object Reconstruction",
    description="Upload a test fMRI data file to visualize the reconstructed 3D shape interactively."
)

if __name__ == '__main__':
    demo.launch(share=True)