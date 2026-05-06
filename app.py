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
    ["examples/example1.png"],
    ["examples/example2.png"]
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

        Upload a brain MRI image and the model will automatically segment tumor regions.

        ### Model
        - UNet
        - EfficientNet-B4 Encoder
        - PyTorch
        - Gradio
        """
    )

    with gr.Row():


        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload MRI Image",
                height=400
            )

            predict_btn = gr.Button(
                "Predict Segmentation",
                variant="primary"
            )

        with gr.Column():

            overlay_output = gr.Image(
                type="numpy",
                label="Tumor Overlay",
                height=400
            )

            mask_output = gr.Image(
                type="numpy",
                label="Predicted Mask",
                height=400
            )



    gr.Examples(
        examples=examples,
        inputs=input_image
    )


    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            overlay_output,
            mask_output
        ]
    )


    gr.Markdown(
        """
        ---
        ### Notes

        - Red regions indicate predicted tumor areas.
        - Predictions are generated using a deep learning segmentation model.
        """
    )

# =========================================================
# LAUNCH
# =========================================================

demo.queue()

demo.launch()