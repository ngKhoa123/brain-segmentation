import gradio as gr
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

from albumentations.pytorch import ToTensorV2
import albumentations as A

IMG_SIZE = 256
BEST_TH = 0.3

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = smp.Unet(
    encoder_name="timm-efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load(
        "best.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

transform = A.Compose([

    A.Resize(
        IMG_SIZE,
        IMG_SIZE
    ),

    A.Normalize(
        mean=(0.485,0.456,0.406),
        std =(0.229,0.224,0.225)
    ),

    ToTensorV2()

])

def predict(image):

    if image is None:
        return None, None

    image = np.array(image)

    orig = image.copy()

    h, w = orig.shape[:2]

    aug = transform(image=image)

    x = aug["image"]

    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = torch.sigmoid(
            model(x)
        )[0,0].cpu().numpy()

    mask = (
        pred > BEST_TH
    ).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    color_mask = np.zeros_like(orig)

    color_mask[mask == 1] = [255, 0, 0]

    overlay = cv2.addWeighted(
        orig,
        0.7,
        color_mask,
        0.3,
        0
    )

    mask_vis = (
        mask * 255
    ).astype(np.uint8)

    return overlay, mask_vis

examples = [
    ["examples/example1.jpg"],
    ["examples/example2.jpg"]
]

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray"
)

with gr.Blocks(
    theme=theme,
    title="Brain Tumor Segmentation"
) as demo:

    gr.Markdown(
        """
        # Brain Tumor Segmentation

        UNet + EfficientNet-B4 for MRI tumor segmentation.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            input_image = gr.Image(
                type="pil",
                label="MRI Image",
                height=300
            )

            gr.Examples(
                examples=examples,
                inputs=input_image
            )

            predict_btn = gr.Button(
                "Predict",
                variant="primary"
            )

        with gr.Column(scale=1):

            overlay_output = gr.Image(
                type="numpy",
                label="Overlay",
                height=300
            )

        with gr.Column(scale=1):

            mask_output = gr.Image(
                type="numpy",
                label="Mask",
                height=300
            )

    gr.Markdown(
        """
        Red regions indicate predicted tumor areas.
        """
    )

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            overlay_output,
            mask_output
        ]
    )

demo.queue()

demo.launch()