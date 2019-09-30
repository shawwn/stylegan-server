import sys
import json
import tempfile

import math
import pickle
import PIL.Image
import numpy as np
import config
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

import math
import numpy as np
from glob import glob
model_res = 1024
model_scale = int(2*(math.log(model_res,2)-1))

def generate_raw_image(latent_vector):
    latent_vector = latent_vector.reshape((1, model_scale, 512))
    generator.set_dlatents(latent_vector)
    return generator.generate_images()[0]

def generate_image(latent_vector):
    img_array = generate_raw_image(latent_vector)
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img

def load_latent(fname):
    rep = np.load(fname)
    return np.expand_dims(rep, axis=0)

if __name__ == '__main__':
    latent=np.array(json.loads(sys.stdin.read()))
    print(repr(latent))
    img = generate_image(latent)
    with tempfile.TemporaryDirectory() as d:
        fname=os.path.join(d, "image.jpg")
        img.save(fname, "JPEG")
        with open(fname, "rb") as f:
            imgbytes = f.read()
    print("Generated {} bytes".format(len(imgbytes)))
