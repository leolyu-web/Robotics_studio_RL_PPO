import numpy as np
import os
from PIL import Image
import inspect
import gymnasium as gym
import xml.etree.ElementTree as ET

def create_uneven_terrain(output_dir=".", size=(128, 128), max_height=2, terrain_size=(30, 30)):
    """
    Generates a heightfield PNG and a custom MuJoCo XML with a checkerboard texture.
    """
    image_path_rel = "terrain.png"
    image_path_abs = os.path.abspath(os.path.join(output_dir, image_path_rel))
    xml_path = os.path.join(output_dir, "ant_uneven.xml")

    # --- Step 1: Generate the heightfield image (No change here) ---
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

    # --- Step 2: Create the custom XML file by modifying the original ---
    print(f"Creating custom XML at {xml_path}...")
    gym_path = os.path.dirname(inspect.getfile(gym))
    original_xml_path = os.path.join(gym_path, "envs/mujoco/assets/ant.xml")

    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    # Find worldbody and remove the original floor
    worldbody = root.find("worldbody")
    floor = worldbody.find("./geom[@name='floor']")
    if floor is not None:
        worldbody.remove(floor)
    else:
        raise ValueError("Could not find floor geom in ant.xml")

    # Find the asset block
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(2, asset)

    # --- EDIT START: Modify the material for better tiling on the new terrain ---
    # Find the material that uses the checkerboard texture. In ant.xml, it's 'MatPlane'.
    matplane = asset.find("./material[@name='MatPlane']")
    if matplane is not None:
        # Change the texrepeat attribute. The original value is for a very large plane.
        # A smaller value like "15 15" will create a reasonably sized checker pattern
        # on our 30x30 terrain. You can adjust these numbers to make the squares larger or smaller.
        matplane.set("texrepeat", "15 15")
        print("Modified 'MatPlane' material to repeat texture 15x15 on the new terrain.")
    # --- EDIT END ---

    # Add the hfield definition to the asset block
    ET.SubElement(
        asset, "hfield",
        name="terrain", file=image_path_abs,
        size=f"{terrain_size[0]} {terrain_size[1]} {max_height} 0.1"
    )
    
    # Add the new hfield geom to the worldbody.
    # It will use the modified 'MatPlane' material to show the checkerboard.
    ET.SubElement(
        worldbody, "geom",
        attrib={
            "conaffinity": "1", "condim": "3", "name": "terrain",
            "material": "MatPlane", "type": "hfield", "hfield": "terrain"
        },
    )

    # Write the modified XML to a new file
    tree.write(xml_path, encoding="unicode")
    print(f"Custom XML with checkerboard terrain created at {xml_path}.")

    return os.path.abspath(xml_path)