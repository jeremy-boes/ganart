import os
import re
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])

import imageio

inputPath = '/folder/with/images/from/GAN'
outputPath = '/location/and/name/of/output/GIF'

files = os.listdir(inputPath)

filesICareAbout = [file for file in files if 'hori' not in file]
FILE_REGEX = re.compile(r"_epoch_(\d+)_batch_0.png")
# print(filesICareAbout)
filesICareAbout.sort(key=lambda x: int(FILE_REGEX.match(x).group(1)))
# print(filesICareAbout)

images = [imageio.imread(os.path.join(inputPath, file)) for file in filesICareAbout]
imageio.mimsave(outputPath, images)
