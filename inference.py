import torch
import numpy as np
import cv2
from PIL import Image
import argparse
import os

from model.forkmamba import ForkMamba

def load_image(image_path):
    ext = os.path.splitext(image_path)[-1].lower()
    if ext == ".npy":
        img = np.load(image_path)
        if img.ndim == 3 and img.shape[-1] in [1, 3]:
            img = np.transpose(img, (2, 0, 1))
    else:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1)) 
    return img

def preprocess(img, target_size=(256, 256)):
    # Resize and normalize
    img = np.transpose(img, (1, 2, 0)) 
    img = cv2.resize(img, target_size)
    img = np.transpose(img, (2, 0, 1)) 
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img

def postprocess(mask):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = ForkMamba(in_channels=3, num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    img = load_image(args.image_path)
    img = preprocess(img, target_size=tuple(args.input_size))

    img = img.to(device)
    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred)
    mask = postprocess(pred)

    if args.output_path:
        cv2.imwrite(args.output_path, mask)
        print(f"Saved mask to {args.output_path}")
    else:
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image (.npy, .jpg, .png, ...)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights (best_model.pth)")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save the output mask")
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 256], help="Input size (H W) for the model")
    args = parser.parse_args()
    main(args)

# python inference.py --image_path path/to/image.png --model_path checkpoints/best_model.pth --output_path mask.png