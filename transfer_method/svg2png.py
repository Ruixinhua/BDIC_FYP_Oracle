# import cairosvg
import cv2
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import os


top_dir = "C:\\Users\\Rui\\Documents\\dataset\\raw_data"
new_root_dir = "C:\\Users\\Rui\\Documents\\dataset\\unpaired"
index = 0
for root, dirs, files in os.walk(top_dir, topdown=False):
    if not dirs:
        if not root.endswith("说文解字的篆字\svg"):
            continue
        print("Current root directory %s" % root)
        sub_dirs = root.split(os.sep)[-3:]
        for f in files:
            new_path = os.path.join(new_root_dir, "zhuan", sub_dirs[0], str(index))
            index += 1
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            svg_file = svg2rlg(os.path.join(root, f))
            new_image_path = os.path.join(new_path, f.replace("svg", "png"))
            renderPM.drawToFile(svg_file, new_image_path)
            pic = Image.open(new_image_path)
            pic = pic.resize((96, 96))
            pic.save(new_image_path, "png")
            print("Convert %s" % f)
