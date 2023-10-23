from typing import List
import math
import os

import numpy as np
import torch
import einops
import pytorch_lightning as pl
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from openxlab.model import download
from tqdm import tqdm

from model.spaced_sampler import SpacedSampler
from model.cldm import ControlLDM
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from utils.face_restoration_helper import FaceRestoreHelper


# download models to local directory
download(model_repo="linxinqi/DiffBIR", model_name="diffbir_general_full_v1")
download(model_repo="linxinqi/DiffBIR", model_name="diffbir_general_swinir_v1")
download(model_repo="linxinqi/DiffBIR", model_name="diffbir_face_full_v1")

config = "cldm.yaml"
general_full_ckpt = "general_full_v1.ckpt"
general_swinir_ckpt = "general_swinir_v1.ckpt"
face_full_ckpt = "face_full_v1.ckpt"

# create general model
general_model: ControlLDM = instantiate_from_config(OmegaConf.load(config)).cuda()
load_state_dict(general_model, torch.load(general_full_ckpt, map_location="cuda"), strict=True)
load_state_dict(general_model.preprocess_model, torch.load(general_swinir_ckpt, map_location="cuda"), strict=True)
general_model.freeze()

# keep a reference of general model's preprocess model and parallel model
general_preprocess_model = general_model.preprocess_model
general_control_model = general_model.control_model

# create face model
face_model: ControlLDM = instantiate_from_config(OmegaConf.load(config))
load_state_dict(face_model, torch.load(face_full_ckpt, map_location="cpu"), strict=True)
face_model.freeze()

# share the pretrained weights with general model
_tmp = face_model.first_stage_model
face_model.first_stage_model = general_model.first_stage_model
del _tmp

_tmp = face_model.cond_stage_model
face_model.cond_stage_model = general_model.cond_stage_model
del _tmp

_tmp = face_model.model
face_model.model = general_model.model
del _tmp

face_model.cuda()

def to_tensor(image, device, bgr2rgb=False):
    if bgr2rgb:
        image = image[:, :, ::-1]
    image_tensor = torch.tensor(image[None] / 255.0, dtype=torch.float32, device=device).clamp_(0, 1)
    image_tensor = einops.rearrange(image_tensor, "n h w c -> n c h w").contiguous()
    return image_tensor

def to_array(image):
    image = image.clamp(0, 1)
    image_array = (einops.rearrange(image, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    return image_array

@torch.no_grad()
def process(
    control_img: Image.Image,
    use_face_model: bool,
    num_samples: int,
    sr_scale: int,
    disable_preprocess_model: bool,
    strength: float,
    positive_prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    steps: int,
    use_color_fix: bool,
    seed: int,
    tiled: bool,
    tile_size: int,
    tile_stride: int
    # progress = gr.Progress(track_tqdm=True)
) -> List[np.ndarray]:
    pl.seed_everything(seed)
    
    global general_model
    global face_model
    
    model = general_model
    sampler = SpacedSampler(model, var_type="fixed_small")
    model.control_scales = [strength] * 13
    if use_face_model:
        print("use face model")
        sampler_face = SpacedSampler(face_model, var_type="fixed_small")
        face_model.control_scales = [strength] * 13
    
    # prepare condition
    if sr_scale != 1:
        control_img = control_img.resize(
            tuple(math.ceil(x * sr_scale) for x in control_img.size),
            Image.BICUBIC
        )
    input_size = control_img.size
    if not tiled:
        control_img = auto_resize(control_img, 512)
    else:
        control_img = auto_resize(control_img, tile_size)
    h, w = control_img.height, control_img.width
    control_img = pad(np.array(control_img), scale=64) # HWC, RGB, [0, 255]

    if use_face_model:
        # set up FaceRestoreHelper
        face_size = 512
        face_helper = FaceRestoreHelper(device=model.device, upscale_factor=1, face_size=face_size, use_parse=True)
        # read BGR numpy [0, 255]
        face_helper.read_image(np.array(control_img)[:, :, ::-1])
        # detect faces in input lq control image
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()

    control = to_tensor(control_img, device=model.device)
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    height, width = control.size(-2), control.size(-1)
    
    preds = []
    for _ in tqdm(range(num_samples)):
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
        if not tiled:
            samples = sampler.sample(
                steps=steps, shape=shape, cond_img=control,
                positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                cfg_scale=cfg_scale, cond_fn=None,
                color_fix_type="wavelet" if use_color_fix else "none"
            )
        else:
            samples = sampler.sample_with_mixdiff(
                tile_size=int(tile_size), tile_stride=int(tile_stride),
                steps=steps, shape=shape, cond_img=control,
                positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                cfg_scale=cfg_scale, cond_fn=None,
                color_fix_type="wavelet" if use_color_fix else "none"
            )
        restored_bg = to_array(samples)

        if use_face_model and len(face_helper.cropped_faces) > 0:
            shape_face = (1, 4, face_size // 8, face_size // 8)
            x_T_face = torch.randn(shape_face, device=model.device, dtype=torch.float32)
            # face detected
            for cropped_face in face_helper.cropped_faces:
                cropped_face = to_tensor(cropped_face, device=model.device, bgr2rgb=True)
                if not disable_preprocess_model:
                    cropped_face = face_model.preprocess_model(cropped_face)
                samples_face = sampler_face.sample(
                    steps=steps, shape=shape, cond_img=cropped_face,
                    positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T_face,
                    cfg_scale=1.0, cond_fn=None,
                    color_fix_type="wavelet" if use_color_fix else "none"
                )
                restored_face = to_array(samples_face)
                face_helper.add_restored_face(restored_face[0])
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=restored_bg[0]
            )

        # remove padding and resize to input size
        restored_img = Image.fromarray(restored_img[:h, :w, :]).resize(input_size, Image.LANCZOS)
        preds.append(np.array(restored_img))
    return preds

MAX_SIZE = int(os.getenv("MAX_SIZE"))
CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT"))

print(f"max size = {MAX_SIZE}, concurrency_count = {CONCURRENCY_COUNT}")

MARKDOWN = \
"""
## DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

[GitHub](https://github.com/XPixelGroup/DiffBIR) | [Paper](https://arxiv.org/abs/2308.15070) | [Project Page](https://0x3f3f3f3fun.github.io/projects/diffbir/)

If DiffBIR is helpful for you, please help star the GitHub Repo. Thanks!

## NOTE

1. This app processes user-uploaded images in sequence, so it may take some time before your image begins to be processed.
2. This is a publicly-used app, so please don't upload large images (>= 1024) to avoid taking up too much time.
"""

block = gr.Blocks().queue(concurrency_count=CONCURRENCY_COUNT, max_size=MAX_SIZE)
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Options", open=True):
                use_face_model = gr.Checkbox(label="Use Face Model", value=False)
                tiled = gr.Checkbox(label="Tiled", value=False)
                tile_size = gr.Slider(label="Tile Size", minimum=512, maximum=1024, value=512, step=256)
                tile_stride = gr.Slider(label="Tile Stride", minimum=256, maximum=512, value=256, step=128)
                num_samples = gr.Slider(label="Number Of Samples", minimum=1, maximum=12, value=1, step=1)
                sr_scale = gr.Number(label="SR Scale", value=1)
                positive_prompt = gr.Textbox(label="Positive Prompt", value="")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                )
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set to a value larger than 1 to enable it!)", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                disable_preprocess_model = gr.Checkbox(label="Disable Preprocess Model", value=False)
                use_color_fix = gr.Checkbox(label="Use Color Correction", value=True)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=231)
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(height="auto", grid=2)
            # gr.Markdown("## Image Examples")
            gr.Examples(
                examples=[
                    ["examples/face/0229.png", True, 1, 1, False, 1.0, "", "", 1.0, 50, True, 231, False, 512, 256],
                    ["examples/face/hermione.jpg", True, 1, 2, False, 1.0, "", "", 1.0, 50, True, 231, False, 512, 256],
                    ["examples/general/14.jpg", False, 1, 4, False, 1.0, "", "", 1.0, 50, True, 231, False, 512, 256],
                    ["examples/general/49.jpg", False, 1, 4, False, 1.0, "", "", 1.0, 50, True, 231, False, 512, 256],
                    ["examples/general/53.jpeg", False, 1, 4, False, 1.0, "", "", 1.0, 50, True, 231, False, 512, 256],
                    # ["examples/general/bx2vqrcj.png", False, 1, 4, False, 1.0, "", "", 1.0, 50, True, 231, True, 512, 256],
                ],
                inputs=[
                    input_image,
                    use_face_model,
                    num_samples,
                    sr_scale,
                    disable_preprocess_model,
                    strength,
                    positive_prompt,
                    negative_prompt,
                    cfg_scale,
                    steps,
                    use_color_fix,
                    seed,
                    tiled,
                    tile_size,
                    tile_stride
                ],
                outputs=[result_gallery],
                fn=process,
                cache_examples=True,
            )
    
    inputs = [
        input_image,
        use_face_model,
        num_samples,
        sr_scale,
        disable_preprocess_model,
        strength,
        positive_prompt,
        negative_prompt,
        cfg_scale,
        steps,
        use_color_fix,
        seed,
        tiled,
        tile_size,
        tile_stride
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch()
