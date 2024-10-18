import os
from PIL import Image

def convert_to_grayscale(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(input_dir, filename)).convert('L')
            img.save(os.path.join(output_dir, filename))

# Caminhos das pastas
rgb_train_dir = './datasets/Leafs/A/train'
gray_train_dir = './datasets/Leafs/B/train'
rgb_val_dir = './datasets/Leafs/A/val'
gray_val_dir = './datasets/Leafs/B/val'
rgb_test_dir = './datasets/Leafs/A/test'
gray_test_dir = './datasets/Leafs/B/test'

# Converter as imagens
convert_to_grayscale(rgb_train_dir, gray_train_dir)
convert_to_grayscale(rgb_test_dir, gray_test_dir)