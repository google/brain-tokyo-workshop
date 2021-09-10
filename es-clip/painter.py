#!/usr/bin/env python3
import numpy as np
from PIL import Image, ImageDraw

class TrianglesPainter(object):

    def __init__(self, h, w, n_triangle=10, alpha_scale=0.1, coordinate_scale=1.0):
        self.h = h
        self.w = w
        self.n_triangle = n_triangle
        self.alpha_scale = alpha_scale
        self.coordinate_scale = coordinate_scale
        
    @property
    def n_params(self):
        return self.n_triangle * 10 # [x0, y0, x1, y1, x2, y2, r, g, b, a]
         
    def random_params(self):
        return np.random.rand(self.n_params)
    
    def render(self, params, background='noise'):
        h, w = self.h, self.w
        alpha_scale = self.alpha_scale
        coordinate_scale = self.coordinate_scale
        
        params = params.reshape(-1, 10).copy()
        
        n_triangle = params.shape[0]
        n_feature = params.shape[1]
        for j in range(n_feature):
            params[:, j] = (params[:, j] - params[:, j].min()) / (params[:, j].max() - params[:, j].min())
        
        if background == 'noise':
            img = Image.fromarray(  (np.random.rand( h, w, 3 ) * 255).astype(np.uint8) )
        elif background == 'white':
            img = Image.new("RGB", (w, h), (255, 255, 255))
        elif background == 'black':
            img = Image.new("RGB", (w, h), (0, 0, 0))
        else:
            assert False
        draw = ImageDraw.Draw(img, 'RGBA')
        
        params = params.tolist()
        for i in range(n_triangle):
            slice_ = params[i]
            
            x0, y0, x1, y1, x2, y2, r, g, b, a = slice_
            xc, yc = (x0 + x1 + x2) / 3. , (y0 + y1 + y2) / 3.
            
            x0, y0 = xc + (x0 - xc) * coordinate_scale, yc + (y0 - yc) * coordinate_scale
            x1, y1 = xc + (x1 - xc) * coordinate_scale, yc + (y1 - yc) * coordinate_scale
            x2, y2 = xc + (x2 - xc) * coordinate_scale, yc + (y2 - yc) * coordinate_scale
            
            x0, x1, x2 = int(x0 * h), int(x1 * h), int(x2 * h)
            y0, y1, y2 = int(y0 * w), int(y1 * w), int(y2 * w)
            r, g, b, a = int(r * 255), int(g * 255), int(b * 255), int(a * alpha_scale * 255)
            
            draw.polygon([(y0, x0), (y1, x1), (y2, x2)], (r, g, b, a))
        
        del draw
        
        img_arr = np.array(img)
        return img_arr