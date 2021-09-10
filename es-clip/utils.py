#!/usr/bin/env python3

import os

import numpy as np

from PIL import Image

def img2arr(img):
    return np.array(img)

def arr2img(arr):
    return Image.fromarray(arr)

def rgba2rgb(rgba_img):
    h, w = rgba_img.size
    rgb_img = Image.new('RGB', (h, w))
    rgb_img.paste(rgba_img)
    return rgb_img

def save_as_gif(fn, imgs, fps=24):
    img, *imgs = imgs
    with open(fn, 'wb') as fp_out:
        img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=int(1000./fps), loop=0)

def save_as_frames(fn, imgs, overwrite=True):
    # save to folder `fn` with sequenced filenames
    os.makedirs(fn, exist_ok=True)
    for i, img in enumerate(imgs):
        this_fn = os.path.join(fn, f'{i:08}.png')
        if overwrite or not os.path.exists(this_fn):
            save_as_png(this_fn, img)

def save_as_png(fn, img):
    if not fn.endswith('.png'):
        fn = f'{fn}.png'
    img.save(fn)

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__  # type: ignore 
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# Copied from https://github.com/makinacorpus/easydict/blob/master/easydict/__init__.py
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)
