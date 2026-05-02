# ForkMamba

This repository contains the implementation of **ForkMamba: Adaptive Multi-branch Visual Mamba with Weighted Concave Region Objective Function for Skin Lesion Segmentation**. It includes data processing, model code, training and inference scripts.

<!-- > **[ForkMamba: Adaptive Multi-branch Visual Mamba with Weighted Concave Region Objective Function for Skin Lesion Segmentation](***url****)**</br>
> Viet-Thanh Nguyen, Thi-Thao Tran, and Van-Truong Pham</br> -->

## Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/vietthanh2710/forkmamba
    cd forkmamba
    ```

2. **Create conda environment**
    ```bash
    conda create -n forkmamba python=3.10
    conda activate forkmamba
    ```

3. **Install mamba-ssm selective_scan_fn (required for VSS modules)**

   **Option 1: Quick installation (recommended)**
    ```bash
    pip install mamba-ssm==2.2.4 --no-build-isolation
    ```

   **Option 2: Follow the original Mamba repository**
   
   Follow the installation guide from the [official Mamba repository](https://github.com/state-spaces/mamba?tab=readme-ov-file#installation) (thanks to Mamba's authors):
    ```bash
    # Install from source
    pip install causal-conv1d>=1.0.0
    pip install mamba-ssm --no-build-isolation
    ```

   **Option 3: Compact alternative for testing (mamba-mini)**
   
   For testing behavior or when GPU is not available, you can use the compact [mamba-mini version](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan) (thanks to VMamba's authors)

4. **Install other dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Training

1. **Prepare your data**  
   - Place your data as a `.npy` file with keys `"image"` and `"mask"`, or modify `/dataset/dataset.py` to fit your data format.

2. **Edit `config.yaml`**  
   - Set hyperparameters such as `epochs`, `batch_size`, `learning_rate`, `loss_type`, etc.

   Example:
   ```yaml
   epochs: 200
   batch_size: 8
   learning_rate: 0.0002
   loss_type: "WCRLoss"
   ```

3. **Run training**

   **Option 1: Using the training script (recommended for distributed training)**
   
   Edit the `train.sh` file to set your configuration:
   ```bash
   # Edit these variables in train.sh
   num_gpus=2                    # Number of GPUs
   port=12345                   # Port for distributed training
   data_path="/path/to/data.npy" # Path to your data
   save_path="/path/to/checkpoints" # Where to save models
   config_path="config.yaml"     # Path to config file
   ```
   
   Then run:
   ```bash
   bash train.sh
   ```

   **Option 2: Direct Python execution**
    ```bash
    python train.py --config config.yaml --data-path /path/to/data.npy --save-path /path/to/checkpoints
    ```

   - The best model will be saved to the directory specified by `save_path`.
   - Supports distributed training with multiple GPUs using `torch.distributed.launch`.


---

## Inference

1. **Run inference on a single image**
    ```bash
    python inference.py --image_path path/to/image.png --model_path checkpoints/best_model.pth --output_path mask.png
    ```

    - Supports `.jpg`, `.png`, and `.npy` images.
    - The output mask will be saved as a PNG file.

2. **Arguments**
    - `--image_path`: Path to the input image (required)
    - `--model_path`: Path to the trained model weights (required)
    - `--output_path`: Path to save the output mask (optional; if not set, mask will be displayed)
    - `--input_size`: Input size for the model, e.g., `--input_size 256 256` (default: 256x256)

---

## Citation

If you use this code for your research, please cite the original paper: 
