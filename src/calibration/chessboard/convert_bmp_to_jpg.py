from PIL import Image
import os
import fnmatch
from pathlib import Path

matches = []
for root, dirnames, filenames in os.walk('chessboard'):
    for filename in fnmatch.filter(filenames, '*.bmp'):
        matches.append(os.path.join(root, filename))
print(matches)

for fname in matches:
    img = Image.open(fname)
    print(fname[:-3])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    new_fname = dir_path + "/jpgs/" + Path(fname).stem + ".jpg"
    img.save(new_fname, "jpeg")
