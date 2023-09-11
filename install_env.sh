python -m pip install --upgrade pip

python gradio_diffbir.py \
--ckpt weights/general_full_v1.ckpt \
--config configs/model/cldm.yaml \
--reload_swinir \
--swinir_ckpt weights/general_swinir_v1.ckpt

pip install pytorch_lightning==1.4.2
pip install einops
conda install transformers
conda install chardet
pip install open-clip-torch
pip install omegaconf
pip install torchmetrics==0.6.0
pip install triton
pip install opencv-python-headless
conda install scipy
conda install matplotlib
pip install lpips
pip install gradio
