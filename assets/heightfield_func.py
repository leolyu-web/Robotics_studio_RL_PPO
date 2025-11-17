import numpy as np
import os
from PIL import Image

def create_uneven_terrain(output_dir=".", size=(128, 128)):
    """
    Generates a heightfield PNG and a custom MuJoCo XML with a black and blue checkerboard texture.
    """
    image_path_rel = "assets/terrain.png"
    image_path_abs = os.path.abspath(os.path.join(output_dir, image_path_rel))
    xml_path = os.path.join(output_dir, "env/ant_uneven.xml")

    # Generate the heightfield image
    print(f"Generating heightfield image at {image_path_abs}...")
    x = np.linspace(-6 * np.pi, 6 * np.pi, size[1])
    y = np.linspace(-6 * np.pi, 6 * np.pi, size[0])
    xx, yy = np.meshgrid(x, y)
    base_wave = np.sin(xx) + np.sin(yy)
    sharp_wave = 0.5 * (np.sin(2.5 * xx) + np.sin(2.5 * yy))
    z = base_wave + sharp_wave
    height_data = (255 * (z - z.min()) / (z.max() - z.min())).astype(np.uint8)
    img = Image.fromarray(height_data, 'L')
    img.save(image_path_abs)
    print("Heightfield image generated.")

if __name__ == "__main__":
    create_uneven_terrain()