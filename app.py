import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

@st.cache_resource
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location="cpu"))
    model.eval()
    return model

def preprocess(canvas_image: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(canvas_image.astype("uint8"), "RGBA").convert("L")
    arr = np.array(img)

    # Find bounding box of non-zero (drawn) pixels
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if rows.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding proportional to the digit size, at least 2px
        pad = max(2, int(0.1 * max(rmax - rmin, cmax - cmin)))
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1] - 1, cmax + pad)

        cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

        # Fit the cropped digit into a square, then center it in 28x28
        h, w = cropped.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = cropped

        img = Image.fromarray(square).resize((28, 28), Image.LANCZOS)
    else:
        img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    # shape: (1, 1, 28, 28) — batch × channel × H × W
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("MNIST Digit Classifier")
st.write("Draw a digit (0–9) in the canvas below.")

# Inject CSS into the canvas component's iframe to make toolbar buttons visible
st.markdown("""
<script>
(function styleCanvasToolbar() {
    const css = `
        button {
            background-color: #555 !important;
            color: #fff !important;
            border: 1px solid #888 !important;
            border-radius: 4px !important;
            padding: 3px 10px !important;
            margin: 2px !important;
            cursor: pointer !important;
            font-size: 13px !important;
        }
        button:hover { background-color: #777 !important; }
    `;
    function inject() {
        document.querySelectorAll('iframe').forEach(f => {
            try {
                const doc = f.contentDocument || f.contentWindow.document;
                if (doc && !doc.head.querySelector('#cc-canvas-fix')) {
                    const s = doc.createElement('style');
                    s.id = 'cc-canvas-fix';
                    s.textContent = css;
                    doc.head.appendChild(s);
                }
            } catch(_) {}
        });
    }
    inject();
    setInterval(inject, 800);
})();
</script>
""", unsafe_allow_html=True)

model = load_model()

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=18,
    stroke_color="#ffffff",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    if canvas_result.image_data[:, :, :3].max() > 10:
        tensor = preprocess(canvas_result.image_data)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().numpy()

        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted]) * 100

        st.markdown(f"### Prediction: **{predicted}** ({confidence:.1f}% confidence)")

        st.bar_chart(
            data={"Confidence": probs},
            x_label="Digit",
            y_label="Probability",
        )
    else:
        st.info("Draw a digit above to see the prediction.")
else:
    st.info("Draw a digit above to see the prediction.")
