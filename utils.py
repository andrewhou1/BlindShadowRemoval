import cv2
import tensorflow as tf
import sys
import glob
import random
import numpy as np
import tensorflow.keras.layers as layers
import math as m

from skimage.draw import line_aa
import matplotlib.tri as mtri
from scipy import ndimage, misc
from PIL import Image, ImageDraw

#arg_scope = tf.contrib.framework.arg_scope 
_MAX_SS_SIGMA = 15  # control subsurface scattering strength  # used to be 10
_MAX_BLUR_SIGMA = 12  # control spatially varying blur strength # used to be 10
_SV_SIGMA = 0.5  # 1. --> not sv blur on boudary; 0. -> always sv blur
lm_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,32,33,34,35,36,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,6,7,8,9,10,11,12,59,58,57,8,9,10,6,7,8,9,10,11,12,59,58,57,8,9,10,6,7,8,9,10,11,12,59,58,57]


def l1_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        #loss = tf.math.reduce_mean(tf.reshape(tf.abs(tf.subtract(x, y)), [xshape[0], -1]), axis=1)
        loss = tf.math.reduce_sum(tf.abs(x-y) * mask) / (tf.reduce_sum(mask) + 1e-6) / x.shape[3]
    else:
        loss = tf.math.reduce_mean(tf.abs(x-y))
    return loss

def l1_loss_yuv(x, y, mask=None):
    pi = tf.constant(m.pi)
    xshape = x.shape
    #rx, gx, bx = tf.split(tf.reverse(x, axis=[-1]),3, axis=3)
    #ry, gy, by = tf.split(tf.reverse(y, axis=[-1]),3, axis=3)
    rx, gx, bx = tf.split(x, 3, axis=3)
    ry, gy, by = tf.split(y, 3, axis=3)
    yx = rx *  .299000 + gx *  .587000 + bx *  .114000
    ux = rx * -.168736 + gx * -.331264 + bx *  .500000
    vx = rx *  .500000 + gx * -.418688 + bx * -.081312
    yy = ry *  .299000 + gy *  .587000 + by *  .114000
    uy = ry * -.168736 + gy * -.331264 + by *  .500000
    vy = ry *  .500000 + gy * -.418688 + by * -.081312
    if mask is not None:
      y_loss = tf.math.reduce_sum(tf.abs(yx-yy) * mask) / (tf.reduce_sum(mask) + 1e-6)
      u_loss = tf.math.reduce_sum(tf.abs(ux-uy) * mask) / (tf.reduce_sum(mask) + 1e-6)
      v_loss = tf.math.reduce_sum(tf.abs(vx-vy) * mask) / (tf.reduce_sum(mask) + 1e-6)
    else:
      y_loss = tf.math.reduce_mean(tf.abs(yx-yy))
      u_loss = tf.math.reduce_mean(tf.abs(ux-uy))
      v_loss = tf.math.reduce_mean(tf.abs(vx-vy))
    return (y_loss+u_loss+v_loss)/2

def l1_loss_hsv(x, y, mask=None):
    pi = tf.constant(m.pi)
    xshape = x.shape
    hx, sx, vx = tf.split(tf.image.rgb_to_hsv(tf.reverse(x, axis=[-1])),3, axis=3)
    hy, sy, vy = tf.split(tf.image.rgb_to_hsv(tf.reverse(y, axis=[-1])),3, axis=3)
    if mask is not None:
      h_loss = tf.math.reduce_sum(tf.abs(tf.cos(2*pi*hx) - tf.cos(2*pi*hy))* mask)/ (tf.reduce_sum(mask) + 1e-6)
      s_loss = tf.math.reduce_sum(tf.abs(sx-sy) * mask) / (tf.reduce_sum(mask) + 1e-6)
      v_loss = tf.math.reduce_sum(tf.abs(vx-vy) * mask) / (tf.reduce_sum(mask) + 1e-6)
    else:
      h_loss = tf.math.reduce_mean(tf.abs(tf.cos(2*pi*hx) - tf.cos(2*pi*hy)))
      s_loss = tf.math.reduce_mean(tf.abs(sx-sy))
      v_loss = tf.math.reduce_mean(tf.abs(vx-vy))
    return (h_loss+v_loss)/2 #(+s_loss+v_loss)/3

def l2_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.math.reduce_sum(tf.square(tf.subtract(x, y)) * mask) / (tf.reduce_sum(mask) + 1e-6) / x.shape[3]
    else:
        loss = tf.math.reduce_mean(tf.square(tf.subtract(x, y)))
    return loss

def l2_loss_yuv(x, y, mask=None):
    pi = tf.constant(m.pi)
    xshape = x.shape
    #rx, gx, bx = tf.split(tf.reverse(x, axis=[-1]),3, axis=3)
    #ry, gy, by = tf.split(tf.reverse(y, axis=[-1]),3, axis=3)
    rx, gx, bx = tf.split(x, 3, axis=3)
    ry, gy, by = tf.split(y, 3, axis=3)
    yx = rx *  .299000 + gx *  .587000 + bx *  .114000
    ux = rx * -.168736 + gx * -.331264 + bx *  .500000
    vx = rx *  .500000 + gx * -.418688 + bx * -.081312
    yy = ry *  .299000 + gy *  .587000 + by *  .114000
    uy = ry * -.168736 + gy * -.331264 + by *  .500000
    vy = ry *  .500000 + gy * -.418688 + by * -.081312
    if mask is not None:
      y_loss = tf.math.reduce_sum(tf.square(tf.subtract(yx, yy)) * mask) / (tf.reduce_sum(mask) + 1e-6)
      u_loss = tf.math.reduce_sum(tf.square(tf.subtract(ux, uy)) * mask) / (tf.reduce_sum(mask) + 1e-6)
      v_loss = tf.math.reduce_sum(tf.square(tf.subtract(vx, vy)) * mask) / (tf.reduce_sum(mask) + 1e-6)
    else:
      y_loss = tf.math.reduce_mean(tf.square(tf.subtract(yx, yy)))
      u_loss = tf.math.reduce_mean(tf.square(tf.subtract(ux, uy)))
      v_loss = tf.math.reduce_mean(tf.square(tf.subtract(vx, vy)))
    return (y_loss+u_loss+v_loss)/2

def hinge_loss(y_pred, y_true, mask=None):
    #y_pred = tf.reshape(y_pred,[y_pred.shape[0], -1])
    return tf.math.reduce_mean(tf.math.maximum(0., 1. - y_true*y_pred))

def style_content_loss(func, inputs):
    inputs = tf.keras.applications.vgg19.preprocess_input(inputs*255)
    style_outputs = func(inputs)
    #style_outputs = func(inputs * 255)
    style_loss = 0
    style_weight = [1.,1.,1.,1.,1.]
    #style_weight = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1/5.6, 6.67]
    for feat, w in zip(style_outputs, style_weight):
      real, fake = tf.split(feat, 2, axis=0)
      style_loss += w* tf.reduce_mean(tf.abs(real-fake))
    return style_loss

def find_edge(mask):
  edge = tf.cast(tf.greater(tf.reduce_mean(mask, axis=3, keepdims=True), .01), tf.float32) - tf.cast(tf.greater(tf.reduce_min(mask, axis=3, keepdims=True), .3), tf.float32)
  rep = 2
  #rep = 3
  kernel = tf.ones((5,5,1))
  #kernel = tf.ones((7,7,1))
  for _ in range(rep):
    edge = tf.nn.dilation2d(edge, kernel, [1,1,1,1], 'SAME', 'NHWC', [1,1,1,1]) 
    edge -= tf.ones_like(edge)
  return tf.cast(tf.greater(edge, 0.), tf.float32)

class Logging(object):
  def __init__(self, config):
    self.config = config
    self.losses = {}
    self.losses_val = {}
    self.txt = ''
    self.fig = []
    self.fig_val = []

  def update(self, losses, training):
    if training:
      for name in losses.keys():
          if name in self.losses:
              current_loss = self.losses[name]
              self.losses[name] = [current_loss[0]+losses[name], current_loss[1]+1]
          else:
              self.losses[name] = [losses[name], 1]
    else:
      for name in losses.keys():
          if name in self.losses_val:
              current_loss = self.losses_val[name]
              self.losses_val[name] = [current_loss[0]+losses[name].numpy(), current_loss[1]+1]
          else:
              self.losses_val[name] = [losses[name].numpy(), 1]

  def display(self, losses, epoch, step, training, allstep):
    self.update(losses, training)
    if training:
        text = 'Epoch (Train) '+str(epoch+1)+'-'+str(step+1)+'/'+str(allstep) + ': '
        for _name in self.losses.keys():
            value = self.losses[_name]
            text += _name+':'+"{:.3g}".format(value[0]/value[1])+', '
    else:
        text = 'Epoch ( Val ) '+str(epoch+1)+'-'+str(step+1)+'/'+str(allstep) + ': '
        for _name in self.losses_val.keys():
            value = self.losses_val[_name]
            text += _name+':'+"{:.3g}".format(value[0]/value[1])+', '
    
    text = text[:-2]+'     '
    # display loss
    print(text, end='\n')
    #print(text, end='\r')
    self.txt = text
    self.epoch = epoch
    self.step = step

  def save(self, fig, training):
    config = self.config
    step = self.step
    fig = self.get_figures(fig)
    if training:
        if step % config.IMG_LOG_FR == 0:
          fname = config.CHECKPOINT_DIR + '/epoch-' + str(self.epoch+1) + '-Train-' + str(self.step+1) + '.png'
          cv2.imwrite(fname, fig.numpy())
        if step % config.TXT_LOG_FR == 0:
          file_object = open(config.CHECKPOINT_DIR+'/log.txt', 'a')
          file_object.write(self.txt+'\n')
          file_object.close()
    else:
        if step % (config.IMG_LOG_FR//10) == 0:
          fname = config.CHECKPOINT_DIR + '/epoch-' + str(self.epoch+1) + '-Val-' + str(self.step+1) + '.png'
          cv2.imwrite(fname, fig.numpy())
        if step % (config.TXT_LOG_FR//10) == 0:
          file_object = open(config.CHECKPOINT_DIR+'/log.txt', 'a')
          file_object.write(self.txt+'\n')
          file_object.close()
    self.fig = []
    self.fig_val = []

  def save_img(self, fig, fname):
    config = self.config
    step = self.step
    fig = self.get_imgs(fig,256)
    fname = config.CHECKPOINT_DIR+'/test/'+fname.split('/')[-2]+'_'+fname.split('/')[-1].split('.')[0]+'-result.png'
    print(fname)
    cv2.imwrite(fname, fig.numpy())
    self.fig = []
    self.fig_val = []

  def reset(self):
    losses = {}
    losses_val = {}
    ind = 0
    for _name in self.loss_names:
      self.losses[_name] = [0, 0]
      self.losses_val[_name] = [0, 0]
      ind += 1
    self.txt = ''
    self.img = 0

  def get_imgs(self, fig, size=None):
    config = self.config
    column = []
    for _img in fig:
      _img = tf.clip_by_value(_img, 0.0, 1.0)*255
      if _img.shape[3] == 1:
          _img = tf.concat([_img, _img, _img], axis=3)
      else:
          r, g, b = tf.split(_img[:,:,:,:3], 3, 3)
          _img = tf.concat([b,g,r], 3)
      if size is None:
        _img = tf.image.resize(_img, [config.FIG_SIZE, config.FIG_SIZE])
      else:
        _img = tf.image.resize(_img, [config.IMG_SIZE, config.IMG_SIZE])
      column.append(_img[0,:,:,:])
    column = tf.concat(column, axis=1)
    return column

  def get_figures(self, fig, size=None):
    config = self.config
    column = []
    for _img in fig:
      _img = tf.clip_by_value(_img, 0.0, 1.0)*255
      if _img.shape[3] == 1:
          _img = tf.concat([_img, _img, _img], axis=3)
      else:
          r, g, b = tf.split(_img[:,:,:,:3], 3, 3)
          _img = tf.concat([b,g,r], 3)
      if size is None:
        _img = tf.image.resize(_img, [config.FIG_SIZE, config.FIG_SIZE])
      else:
        _img = tf.image.resize(_img, [config.IMG_SIZE, config.IMG_SIZE])
      _row = tf.split(_img, _img.shape[0])
      _row = tf.concat(_row, axis=2)
      column.append(_row[0,:,:,:])
    column = tf.concat(column, axis=0)
    return column

def generate_face_region(source, img_size):
    morelm = np.copy(source[0:17,:])
    morelm[:,1] = morelm[0,1] - (morelm[:,1] - morelm[0,1]) * 0.8
    source = np.concatenate([source,morelm],axis=0)
    '''
    img = Image.new('L', (img_size, img_size), 0)
    ImageDraw.Draw(img).polygon(source, outline=1, fill=1)
    mask = np.array(img)
    mask = cv2.GaussianBlur(mask,(5,5),0).reshape([img_size,img_size,1])

    '''
    xi, yi = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    # interp2d
    _triang = mtri.Triangulation(source[:,0], source[:,1])
    _interpx = mtri.LinearTriInterpolator(_triang, source[:,0])
    _offsetmapx = _interpx(xi, yi)

    offsetmap = np.stack([_offsetmapx], axis=2)
    offsetmap = np.nan_to_num(offsetmap)  
    offsetmap = np.asarray(offsetmap>0,np.float32)
    offsetmap = cv2.GaussianBlur(offsetmap,(5,5),0).reshape([img_size,img_size,1])
    return offsetmap

def generate_face_region2(source, imx, imy):
    morelm = np.copy(source[0:17,:])
    morelm[:,1] = morelm[0,1] - (morelm[:,1] - morelm[0,1]) * 0.6
    source = np.concatenate([source,morelm],axis=0)
    xi, yi = np.meshgrid(np.linspace(0, 1, imx), np.linspace(0, 1, imy))

    # interp2d
    _triang = mtri.Triangulation(source[:,0], source[:,1])
    _interpx = mtri.LinearTriInterpolator(_triang, source[:,0])
    _offsetmapx = _interpx(xi, yi)

    offsetmap = np.stack([_offsetmapx], axis=2)
    offsetmap = np.nan_to_num(offsetmap)  
    offsetmap = np.asarray(offsetmap>0,np.float32)
    offsetmap = cv2.blur(offsetmap,(45,45),0).reshape([imy,imx,1])
    offsetmap = offsetmap / (np.max(offsetmap)+1e-6)
    return offsetmap

def generate_landmark_map(landmark, img_size):
    lmlist = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],
              [18,19],[19,20],[20,21],[21,22],[23,24],[24,25],[25,26],[26,27],
              [37,38],[38,39],[39,40],[40,41],[41,42],[42,37],[43,44],[44,45],[45,46],[46,47],[47,48],[48,43],
              [28,29],[29,30],[30,31],[32,33],[33,34],[34,35],[35,36],
              [49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,60],[60,49],
              [61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,68],[68,61]]
    lm_map = []
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    lm = landmark*img_size
    for pr in lmlist:
        lm_start = lm[pr[0]-1,:].astype(np.int32)
        lm_end = lm[pr[1]-1,:].astype(np.int32)
        rr, cc, val = line_aa(lm_start[0], lm_start[1], lm_end[0], lm_end[1])
        templist = [t for t in range(len(rr)) if rr[t] < img_size and rr[t] > 0 ]
        rr = rr[templist]
        cc = cc[templist]
        val = val[templist]
        templist = [t for t in range(len(cc)) if cc[t] < img_size and cc[t] > 0 ]
        rr = rr[templist]
        cc = cc[templist]
        val = val[templist]
        img[cc, rr] = val * 255
    blur = cv2.GaussianBlur(img,(3,3),0)
    blur = blur / np.amax(blur) * 255
    lm_map = np.reshape(blur, [blur.shape[0], blur.shape[1], 1])
    return lm_map

def list_concat(a,b):
    c = []
    for ia, ib in zip(a,b):
        ic = tf.concat([ia,ib], axis=0)
        c.append(ic)
    return c 

def list_split(a, num):
    b = []
    c = []
    for ia in a:
        ib, ic = tf.split(ia, num, axis=0)
        b.append(ib)
        c.append(ic)
    return b, c

def pts_load(path):
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return np.asarray(points, dtype=np.float32)

def face_crop_and_resize(img0, lm0, fsize, box_perturb=[1.15, 1.25], aug=False):
    img = np.copy(img0)
    lm  = np.copy(lm0)
    img_shape = img.shape
    lm_reverse_list = np.array([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,
                               27,26,25,24,23,22,21,20,19,18,
                               28,29,30,31,36,35,34,33,32,
                               46,45,44,43,48,47,40,39,38,37,42,41,
                               55,54,53,52,51,50,49,60,59,58,57,56,65,64,63,62,61,68,67,66],np.int32) -1                
    if aug and random.uniform(0,1)>0.5:
        #img = cv2.flip(img, 1)
        #lm[:,0] = img_shape[1] - lm[:,0]
        #lm = lm[lm_reverse_list,:]
        # rotate
        rot   = np.random.uniform(low=-10, high=10)
        sin_rot = np.sin(rot * np.pi / 180.)
        cos_rot = np.cos(rot * np.pi / 180.)
        lm_new = np.copy(lm)
        w, h, _ = img.shape
        lm[:,0] -= h/2
        lm[:,1] -= w/2
        lm_new[:,0] = lm[:,1] * sin_rot + lm[:,0] * cos_rot
        lm_new[:,1] = lm[:,1] * cos_rot - lm[:,0] * sin_rot
        lm_new[:,0] += h/2
        lm_new[:,1] += w/2
        img = ndimage.rotate(img, rot, reshape=False)
        lm = np.copy(lm_new)
    lm_mirror = np.copy(lm)
    lm_mirror[:,0] = img_shape[1] - lm_mirror[:,0]
    lm_mirror = lm_mirror[lm_reverse_list,:]

    center = [(np.min(lm[:,0])+np.max(lm[:,0]))/2, (np.min(lm[:,1])+np.max(lm[:,1]))/2]
    length = np.max([(np.max(lm[:,0])-np.min(lm[:,0]))/2, (np.max(lm[:,1])-np.min(lm[:,1]))/2]) * 1.4
    if aug:
      center[0] = center[0] + random.uniform(-0.1,0.1)*length
      center[1] = center[1] + random.uniform(-0.1,0.1)*length
      length = length * random.uniform(0.9,1.1)
    box = [int(center[0])-int(length),
           int(center[1])-int(length*1.2),
           int(center[0])+int(length),
           int(center[1])+int(length)+int(length)-int(length*1.2)]
    box0 = [int(center[0])-int(length),
           int(center[1])-int(length*1.2),
           int(center[0])+int(length),
           int(center[1])+int(length)+int(length)-int(length*1.2)]
    #print(box, img.shape)
    box_m = [img_shape[1] - box[2],
             box[1],
             img_shape[1] - box[0],
             box[3]]

    lm[:,0] = lm[:,0] - box[0]
    lm[:,1] = lm[:,1] - box[1]
    lm_mirror[:,0] = lm_mirror[:,0] - box_m[0]
    lm_mirror[:,1] = lm_mirror[:,1] - box_m[1]

    preset_x = 0
    preset_y = 0
    if box[0] < 0 or box[2] > img_shape[1]:
        preset_x = max(-box[0], box[2] - img_shape[1])
    if box[1] < 0 or box[3] > img_shape[0]:
        preset_y = max(-box[1], box[3] - img_shape[0])
    if preset_x > 0 or preset_y > 0:
        img_large= np.zeros((img_shape[0]+preset_y+preset_y+2,img_shape[1]+preset_x+preset_x+2,img_shape[2]))
        img_large[preset_y:preset_y+int(img_shape[0]),preset_x:preset_x+int(img_shape[1]),:] = img
        img = img_large
        box[0] = box[0] + preset_x
        box[1] = box[1] + preset_y
        box[2] = box[2] + preset_x
        box[3] = box[3] + preset_y
    img = img[box[1]:box[3],box[0]:box[2],:]
    sz = img.shape[0]
    if img.shape[0] == img.shape[1] and img.shape[0]>0:
        img = cv2.resize(img,   (fsize,fsize))
    else:
        img = np.zeros((fsize, fsize, img.shape[2])) 

    return img, lm/(length*2), lm_mirror/(length*2), box0 

"""
  Color jitter
"""
def getbias(x, bias):
  """Bias in Ken Perlin’s bias and gain functions."""
  return x / ((1.0 / bias - 2.0) * (1.0 - x) + 1.0 + 1e-6)

def apply_tone_curve(image, gain=(0.5, 0.5, 0.5), is_rgb=False):
  """Apply tone perturbation to images.

  Tone curve jitter comes from Schlick's bias and gain.
  Schlick, Christophe. "Fast alternatives to Perlin’s bias and gain functions." Graphics Gems IV 4 (1994).
  Args:
    image: a 3D image tensor [H, W, C].
    gain: a tuple of length 3 that specifies the strength of the jitter per color channel.
    is_rgb: a bool that indicates whether input is grayscale (C=1) or rgb (C=3).
  
  Returns:
    3D tensor applied with a tone curve jitter, has the same size as input.
  """
  image_max = np.max(image)
  image = (image / (image_max+1e-6)).astype(np.float32)
  if not is_rgb:
    mask = tf.cast(tf.greater_equal(image, 0.5), image.dtype)
    image = getbias(image * 2.0, gain[0]) / 2.0 * (1.0 - mask) + (
        getbias(image * 2.0 - 1.0, 1.0 - gain[0]) / 2.0 + 0.5) * mask
  else:
    image_r = image[..., 0]
    image_r_mask = (image_r>0.499).astype(np.float32)
    image_r = getbias(image_r * 2.0, gain[0]) / 2.0 * (1.0 - image_r_mask) + (
        getbias(image_r * 2.0 - 1.0, 1.0 - gain[0]) / 2.0 + 0.5) * image_r_mask

    image_g = image[..., 1]
    image_g_mask = (image_g>0.499).astype(np.float32)
    image_g = getbias(image_g * 2.0, gain[1]) / 2.0 * (1.0 - image_g_mask) + (
        getbias(image_g * 2.0 - 1.0, 1.0 - gain[1]) / 2.0 + 0.5) * image_g_mask

    image_b = image[..., 2]
    image_b_mask = (image_b>0.499).astype(np.float32)
    image_b = getbias(image_b * 2.0, gain[2]) / 2.0 * (1.0 - image_b_mask) + (
        getbias(image_b * 2.0 - 1.0, 1.0 - gain[2]) / 2.0 + 0.5) * image_b_mask

    image = np.stack([image_r, image_g, image_b], axis=2)
  return image * image_max

def get_ctm_ls_inv(image, target):
  """Use least square to obtain color transfer matrix.

  Args:
    image: the source tensor of shape [H, W, 3].
    target: target tensor with the same shape as input.
  
  Returns:
    tensor of size 3 by 3 that minimizes |C x image - target|_2.
  """
  image = image.reshape(-1,3)
  target = target.reshape(-1,3)
  #ctm = tf.linalg.lstsq(image, target, l2_regularizer=0.0, fast=True)
  ctm = np.linalg.lstsq(image, target, rcond=None)[0]
  ctm_inv = np.linalg.lstsq(target, image, rcond=None)[0]
  return ctm.T, ctm_inv.T

def get_ctm_ls(image, target):
  """Use least square to obtain color transfer matrix.

  Args:
    image: the source tensor of shape [H, W, 3].
    target: target tensor with the same shape as input.
  
  Returns:
    tensor of size 3 by 3 that minimizes |C x image - target|_2.
  """
  image = image.reshape(-1,3)
  target = target.reshape(-1,3)
  #ctm = tf.linalg.lstsq(image, target, l2_regularizer=0.0, fast=True)
  ctm = np.linalg.lstsq(image, target, rcond=None)[0]
  #ctm_inv = np.linalg.lstsq(target, image, rcond=None)[0]
  return ctm.T#, ctm_inv.T


def apply_ctm(image, ctm):
  """Apply a color transfer matrix.

  Args:
    image: a tensor that contains the source image of shape [H, W, 3].
    ctm: a tensor that contains a 3 by 3 color matrix.
  Returns:
    a tensor of the same shape as image.
  """
  shape = image.shape
  image = image.reshape(-1, 3)
  image = np.tensordot(image, ctm, axes=[[-1], [-1]])
  return image.reshape(shape)

def fft_filter(img, kernel):
  """Apply FFT to a 2D tensor.
  Args:
    img: a 2D tensor of the input image [H, W].
    kernel: a 2D tensor of the kernel.
  
  Returns:
    a 2D tensor applied with a filter using FFT.
  """
  with tf.name_scope('fft2d_gray'):
    img = tf.cast(img, tf.complex64)
    kernel = tf.cast(kernel, tf.complex64)
    img_filtered = tf.cast(
        tf.abs(tf.signal.ifft2d(tf.multiply(tf.signal.fft2d(img), tf.signal.fft2d(kernel)))),
        tf.float32)
  return img_filtered


def fft3_filter(img, kernel, is_rgb=True):
  """Apply FFT to a 3D tensor.
  Args:
    img: a 3D tensor of the input image [H, W, C].
    kernel: a 2D tensor of the kernel.
    is_rgb: a bool that indicates whether input is rgb or not.
  
  Returns:
    a filtered 3D tensor, has the same size as input.
  """
  with tf.name_scope('fft2d_rgb'):
    img = tf.cast(img, tf.complex64)
    kernel = tf.cast(kernel, tf.complex64)
  if not is_rgb:
    img_r = fft_filter(img[..., 0], kernel)
    img_r = tf.expand_dims(img_r, 2)
    return img_r
  else:
    img_r = fft_filter(img[..., 0], kernel)
    img_g = fft_filter(img[..., 1], kernel)
    img_b = fft_filter(img[..., 2], kernel)
    img_filtered = tf.stack([img_r, img_g, img_b], 2)
  return img_filtered


def create_disc_filter(r):
  """Create a disc filter of radius r.
  Args:
    r: an int of the kernel radius.
  Returns:
    disk filter: A 2D Tensor
  """
  x, y = tf.meshgrid(tf.range(-r, r + 1), tf.range(-r, r + 1))
  mask = tf.less_equal(tf.pow(x, 2) + tf.pow(y, 2), tf.pow(r, 2))
  mask = tf.cast(mask, tf.float32)
  mask /= tf.reduce_sum(mask)
  return mask

def apply_disc_filter(input_img, kernel_sz, is_rgb=True):
  """Apply disc filtering to the input image with a specified kernel size.
  To handle large kernel sizes, this is operated (and thus approximated) in 
  frequency domain (fft).
  Args:
    input_img: a 2D or 3D tensor. [H, W, 1] or [H, W].
    kernel_sz: a scalar tensor that specifies the disc kernel size.
    is_rgb: a bool that indicates whether FFT is grayscale(c=1) or rgb(c=3).
  Returns:
    A Tensor after applied disc filter, has the same size as the input tensor.
  """
  """
  if kernel_sz == 0:
    print('Input kenrel size is 0.')
    return input_img
  """
  
  disc = create_disc_filter(kernel_sz)
  offset = kernel_sz - 1
  # if len(tf.shape(input_img)) == 2:
  #   padding_img = [[0, kernel_sz], [0, kernel_sz]]
  # elif len(tf.shape(input_img)) == 3:
  padding_img = [[0, kernel_sz], [0, kernel_sz], [0, 0]]
  img_padded = tf.pad(input_img, padding_img, 'constant')
  paddings = [[0, tf.shape(img_padded)[0] - tf.shape(disc)[0]],
              [0, tf.shape(img_padded)[1] - tf.shape(disc)[1]]]
  disc_padded = tf.pad(disc, paddings)
  # if len(tf.shape(input_img)) == 2:
  #   img_blurred = fft_filter(
  #       img_padded, disc_padded)[offset:offset + tf.shape(input_img)[0],
  #                                offset:offset + tf.shape(input_img)[1]]
  # else:
  img_blurred = fft3_filter(
      img_padded, disc_padded,
      is_rgb=is_rgb)[offset:offset + tf.shape(input_img)[0],
                     offset:offset + tf.shape(input_img)[1]]
  return img_blurred

def render_shadow_from_mask(mask, segmentation=None):
  """Render a shadow mask by applying spatially-varying blur.
  Args:
    mask: A Tensor of shape [H, W, 1].
    segmentation: face segmentation, apply to the generated shadow mask if provided.
  Returns:
    A Tensor of shape [H, W, 1] containing the shadow mask.
  """
  mask = tf.expand_dims(mask, 2)
  disc_filter_sz = tf.random.uniform(shape=(), minval=1, maxval=_MAX_BLUR_SIGMA, dtype=tf.int32)
  mask_blurred = tf.cond(tf.greater(tf.random.uniform([]),tf.constant(_SV_SIGMA)), 
        lambda: apply_spatially_varying_blur(mask,
                    blur_size=tf.random.uniform(shape=(), minval=1, maxval=3, dtype=tf.int32)),
        lambda: apply_disc_filter(mask, disc_filter_sz, is_rgb=False))
  mask_blurred_norm = tf.divide(mask_blurred, tf.reduce_max(mask_blurred))
  if segmentation is not None:
    mask_blurred_seg = mask_blurred_norm * segmentation
  else:
    mask_blurred_seg = mask_blurred_norm
  '''
  tf.debugging.assert_greater_equal(
      tf.reduce_sum(mask_blurred_seg),
      0.1,
      message='Rendered silhouette mask values too small.')  # sample drops if this happens'''
  return mask_blurred_norm

def render_perlin_mask(size, segmentation=None):
  """Render a shadow mask using perlin noise pattern.
  Args:
    size: A 2D tensor of target mask size.
    segmentation: face segmentation, apply to the generated shadow mask if provided.
  Returns:
    A Tensor of shape [H, W, 1] containing the shadow mask.
  """
  with tf.name_scope('render_perlin'):
    size = tf.cast(size, tf.int32)
    perlin_map = perlin_collection((size[0], size[1]), [4, 4], 4,
                                   tf.random.uniform([], 0.05, 0.85))
    perlin_map_thre = tf.cast(tf.greater(perlin_map, 0.15), tf.float32)
    perlin_shadow_map = render_shadow_from_mask(
        perlin_map_thre, segmentation=segmentation)
  return perlin_shadow_map


def apply_ss_shadow_map(mask):
  """Apply subsurface scattering approximation to the shadow mask.
  Args:
    mask: A Tensor of shape [H, W, 1].
  Returns:
    A Tensor of shape [H, W, 3] that is applied with wavelength-dependent blur.
  """
  r = tf.random.uniform(
      shape=(), minval=1, maxval=_MAX_SS_SIGMA, dtype=tf.float32)  # a global scalar to scale all the blur size
  shadow_map = wavelength_filter(mask, num_lv=6, scale=r, is_rgb=False)
  shadow_map = tf.minimum(1., shadow_map/0.6)  # a heuristic scalar for more stable normalization
  return shadow_map

def wavelength_filter(input_img, num_lv=6, scale=5, is_rgb=False, name='wavelength_filter'):
  """Image-based subsurface scattering approximation
  Parameters from the NVIDIA screen-space subsurface scattering (SS) slide 98.
  http://developer.download.nvidia.com/presentations/2007/gdc/Advanced_Skin.pdf
  Args:
    input_img: a 3D tensor [H, W, C].
    num_lv: a scalar that specifies the number of Gaussian filter levels in the SS model.
    scale: a scalar that is the scale used to calibrate the kernel size into # pixels based on the size of the face in the image.
    is_rgb: a bool that indicates whether input is grayscale(c=1) or rgb(c=3).
    name: string, name of the graph.
  Returns:
    A 3D tensor after approximated with subsurface scattering.
  """
  with tf.name_scope(name):
    scale = tf.cast(scale, tf.float32)
    ss_weights = np.array([[0.042, 0.22, 0.437, 0.635],
                           [0.220, 0.101, 0.355, 0.365],
                           [0.433, 0.119, 0.208, 0],
                           [0.753, 0.114, 0, 0],
                           [1.412, 0.364, 0, 0],
                           [2.722, 0.080, 0, 0]])
    ss_weights_norm = np.sum(ss_weights, 0)
    img_blur_rgb = 0.
    for lv in range(num_lv):
      if lv != 0:
        blur_kernel = ss_weights[lv, 0] * scale
      else:
        blur_kernel = ss_weights[lv, 0] * scale
      rgb_weights = ss_weights[lv, 1:]
      if not is_rgb:
        blur_img = gaussian_filter(tf.expand_dims(input_img, 0), blur_kernel)[0]
        blur_r = blur_img * rgb_weights[0] * tf.random.uniform([], 1.1, 1.5)
        blur_g = blur_img * rgb_weights[1]
        blur_b = blur_img * rgb_weights[2]
      else:
        blur_r = gaussian_filter(
            tf.expand_dims(input_img[..., 0, tf.newaxis], 0),
            blur_kernel)[0] * rgb_weights[0] * 1. / ss_weights_norm[1]
        blur_g = gaussian_filter(
            tf.expand_dims(input_img[..., 1, tf.newaxis], 0),
            blur_kernel)[0] * rgb_weights[1] * 1. / ss_weights_norm[2]
        blur_b = gaussian_filter(
            tf.expand_dims(input_img[..., 2, tf.newaxis], 0),
            blur_kernel)[0] * rgb_weights[2] * 1. / ss_weights_norm[3]
      img_blur = tf.concat([blur_r, blur_g, blur_b], 2)
      img_blur_rgb += img_blur
  return img_blur_rgb

def gaussian_filter(image, sigma, pad_mode='REFLECT', name='gaussian_filter'):
  """Applies Gaussian filter to an image using depthwise conv.
  Args:
    image: 4-D Tensor with float32 dtype and shape [N, H, W, C].
    sigma: Positive float or 0-D Tensor.
    pad_mode: String, mode argument for tf.pad. Default is 'REFLECT' for
      whole-sample symmetric padding.
    name: A string to name this part of the graph.
  Returns:
    Filtered image, has the same shape with the input.
  """
  with tf.name_scope(name):
    image.shape.assert_has_rank(4)
    sigma = tf.cast(sigma, tf.float32)
    sigma.shape.assert_has_rank(0)  # sigma is a scalar.

    channels = tf.shape(image)[3]
    r = tf.cast(tf.math.ceil(2.0 * sigma), tf.int32)
    n = tf.range(-tf.cast(r, tf.float32), tf.cast(r, tf.float32) + 1)
    coeffs = tf.exp(-0.5 * (n / sigma)**2)
    coeffs /= tf.reduce_sum(coeffs)
    coeffs_x = tf.tile(tf.reshape(coeffs, (1, -1, 1, 1)), (1, 1, channels, 1))
    coeffs_y = tf.reshape(coeffs_x, (2 * r + 1, 1, channels, 1))

    padded = tf.pad(image, ((0, 0), (r, r), (r, r), (0, 0)), pad_mode)
    #with tf.device('/cpu:0'):  # seems necessary for depthwise_conv2d
    filtered = tf.nn.depthwise_conv2d(
        padded, coeffs_x, (1, 1, 1, 1), 'VALID', name='filter_x')
    filtered = tf.nn.depthwise_conv2d(
        filtered, coeffs_y, (1, 1, 1, 1), 'VALID', name='filter_y')
    filtered.set_shape(image.shape)
  return filtered

def get_brightness_mask(size, min_val=0.5):
  """Render per-pixel intensity variation mask within [min_val, 1.].
  Args:
    size: A 2D tensor of target mask size.
  
  Returns:
    A Tensor of shape [H, W, 1] that is generated with perlin noise pattern.
  """
  perlin_map = perlin_collection((size[0], size[1]), [2, 2], 2,
                                 tf.random.uniform([], 0.05, 0.25))
  perlin_map = perlin_map / (1. / (min_val + 1e-6)) + min_val
  perlin_map = tf.minimum(perlin_map, 1.)
  #perlin_map = tf.minimum(perlin_map, 0.6)
  #perlin_map = tf.minimum(perlin_map, 0.8)
  return perlin_map

def perlin_collection(size, reso, octaves, persistence):
  """Generate perlin patterns of varying frequencies.
  Args:
    size: a tuple of the target noise pattern size.
    reso: a tuple that specifies the resolution along lateral and longitudinal.
    octaves: int, number of octaves to use in the perlin model.
    persistence: int, persistence applied to every iteration of the generation.
  
  Returns:
    a 2D tensor of the perlin noise pattern.
  """
  noise = tf.zeros(size)
  amplitude = 1.0

  for _ in range(octaves):
    noise += amplitude * perlin(size, reso)
    amplitude *= persistence
    reso[0] *= 2
    reso[1] *= 2

  return noise

def perlin(size, reso):
  """Generate a perlin noise pattern, with specified frequency along x and y.
  Theory: https://flafla2.github.io/2014/08/09/perlinnoise.html
  Args:
    size: a tuple of integers of the target shape of the noise pattern.
    reso: reso: a tuple that specifies the resolution along lateral and longitudinal (x and y).
  
  Returns:
    a 2D tensor of the target size.
  """
  ysample = tf.linspace(0.0, reso[0], size[0])
  xsample = tf.linspace(0.0, reso[1], size[1])
  xygrid = tf.stack(tf.meshgrid(ysample, xsample), 2)
  xygrid = tf.math.floormod(tf.transpose(xygrid, [1, 0, 2]), 1.0)
  #xygrid = tf.math.floormod(tf.transpose(xygrid, [1, 0, 2]), 1.0)

  xyfade = (6.0 * xygrid**5) - (15.0 * xygrid**4) + (10.0 * xygrid**3)
  angles = 2.0 * np.pi * tf.random.uniform([reso[0] + 1, reso[1] + 1])
  grads = tf.stack([tf.cos(angles), tf.sin(angles)], 2)

  gradone = tf.image.resize(grads[0:-1, 0:-1], [size[0], size[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gradtwo = tf.image.resize(grads[1:, 0:-1],   [size[0], size[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gradthr = tf.image.resize(grads[0:-1, 1:],   [size[0], size[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  gradfou = tf.image.resize(grads[1:, 1:],     [size[0], size[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  gradone = tf.reduce_sum(gradone * tf.stack([xygrid[:, :, 0], xygrid[:, :, 1]], 2), 2)
  gradtwo = tf.reduce_sum(gradtwo * tf.stack([xygrid[:, :, 0] - 1, xygrid[:, :, 1]], 2), 2)
  gradthr = tf.reduce_sum(gradthr * tf.stack([xygrid[:, :, 0], xygrid[:, :, 1] - 1], 2), 2)
  gradfou = tf.reduce_sum(gradfou * tf.stack([xygrid[:, :, 0] - 1, xygrid[:, :, 1] - 1], 2), 2)

  inteone = (gradone * (1.0 - xyfade[:, :, 0])) + (gradtwo * xyfade[:, :, 0])
  intetwo = (gradthr * (1.0 - xyfade[:, :, 0])) + (gradfou * xyfade[:, :, 0])
  intethr = (inteone * (1.0 - xyfade[:, :, 1])) + (intetwo * xyfade[:, :, 1])

  return tf.sqrt(2.0) * intethr


def apply_spatially_varying_blur(image, blur_size=2, blurtype='disk'):
  """Apply spatially-varying blur to an image.
  Using pyramid to approximate for efficiency
  
  Args:
    image: a 3D image tensor [H, W, C].
    blur_size: base value for the blur size in the pyramic.
    blurtype: type of blur, either 'disk' or 'gaussian'.
  
  Returns:
    a 2D tensor of the target size.
  """
  pyramid = create_pyramid(image, blur_size=blur_size, blurtype=blurtype)
  image_blurred = apply_pyramid_blend(pyramid)
  return image_blurred

def lerp(a, b, x):
  """Linear interpolation between a and b using weight x."""
  return a + x * (b - a)


def apply_pyramid_blend(pyramid):
  """Reconstruct an image using bilinear interpolation between pyramid levels.
  Args:
    pyramid: a list of tensors applied with different blur levels.
  
  Returns:
    A reconstructed 3D tensor that is collapsed from the input pyramid.
  """
  num_levels = 3
  guidance_perlin_base = perlin_collection(
      (tf.shape(pyramid[0])[0], tf.shape(pyramid[0])[1]), [2, 2], 1,
      tf.random.uniform([], 0.05, 0.25))
  guidance_perlin_base -= tf.reduce_min(guidance_perlin_base)
  guidance_perlin_base /= tf.reduce_max(guidance_perlin_base)
  guidance_blur = tf.clip_by_value(guidance_perlin_base / (1. / num_levels),
                                   0.0, num_levels)
  image_reconst = pyramid
  for i in range(int(num_levels) - 2, -1, -1):
    alpha = tf.clip_by_value(guidance_blur - i, 0., 1.)
    alpha = tf.expand_dims(alpha, 2)
    image_reconst[i] = lerp(pyramid[i], image_reconst[i + 1], alpha)
  return image_reconst[0]

def create_pyramid(image, blur_size=2, blurtype='disk'):
  """Create a pyramid of different levels of disk blur.
  Args:
    image: a 2D or 3D tensor of the input image.
    blur_size: base value for the blur size in the pyramic.
    blurtype: a string that specifies the kind of blur, either disk or gaussian.
  
  Returns:
    Pyramid: a list of tensors applied with different blur kernels.
  """
  image_pyramid = []
  for i in range(3):
    rsz = np.power(2, i) * blur_size
    if blurtype == 'disk':
      input_lv = apply_disc_filter(image, rsz, is_rgb=False)
    elif blurtype == 'gaussian':
      input_lv = gaussian_filter(tf.expand_dims(input_lv, 0), blur_size)[0, ...]
    else:
      raise ValueError('Unknown blur type.')
    image_pyramid.append(input_lv)
  return image_pyramid

class ShadowMaker():
    #shape match
    #motion:shake-translation-scaling/speed-constant-change/repeat/break_time
    #shape/rotate/scale/flip/blurring
    #inconsistent shadow
    #small shadow --> repeat to be big --> moving window
    #Motion = 'trans' #'trans', 'shake', 'scaling'
    #Speed = 0.1 # 0.1 - 1
    #Scale = 1 # 0.3 - 2
    #Rotation = 0 # 0 - 365
    #Blur = 3 # 0 - 10
    #Inconsistent = True

    def __init__(self, face, lm):
        """Set values of computed attributes."""
        self.mot   = np.random.randint(low=1, high=3) #1 -'trans', 2-'shake', 3-'scaling'
        self.spd_x = np.random.uniform(low=0.1, high=10.0)
        self.spd_y = np.random.uniform(low=0.1, high=10.0)
        self.scale = np.random.uniform(low=1.0, high=2.5)
        self.rot   = np.random.uniform(low=0, high=365.0)
        self.blur  = np.random.randint(low=10, high=15)
        self.incs  = np.random.uniform(size=(1,))
        self.face  = face
        self.lm    = lm
        self.compile_mask()

    def display(self):
        print('*************************************************')
        print('Pattern = '+str(self.mask_type))
        print('Motion ='+str(self.motion))
        print('Speedx ='+str(self.spd_x))
        print('Speedy ='+str(self.spd_y))
        print('Scale ='+str(self.scale))
        print('Rot ='+str(self.rot))
        print('Blur ='+str(self.blur))
        print('Inconsistency ='+str(self.incs))


    def compile_mask(self):
        # generate shadow mask
        _list = glob.glob('/research/cvlshare/cvl-liuyaoj1/Data/shadow/*.png')
        _mask = _list[random.randint(0, len(_list) - 1)]
        mask = cv2.imread(_mask, 0) / 255
        if np.random.uniform(low=0, high=1) > 0.75:
          mask = 1 - mask

        # compute the center 
        lm = self.lm * self.face.shape[0]
        lmp = np.copy(lm[0:17,:])
        lmp[:,1] = lmp[0,1] - (lmp[:,1] - lmp[0,1]) * 0.6
        lm = np.concatenate([lm,lmp],axis=0)

        if _mask[0] != 'm' and np.random.uniform(low=-1, high=1) > 0:
            start_center_idx = np.random.randint(low=17, high=67)
            length = np.max([(np.max(lm[:,0])-np.min(lm[:,0]))/2, (np.max(lm[:,1])-np.min(lm[:,1]))/2])
            start_center = lm[start_center_idx,:]
            start_center[0] = (np.max(lm[:,0])+np.min(lm[:,0]))/2 
            start_center[1] = (np.max(lm[:,1])+np.min(lm[:,1]))/2 
            mask_shape = max(int(length * 2), 10)
            mask = cv2.resize(mask,(mask_shape,mask_shape))
            mask = cv2.blur(mask, (self.blur//2, self.blur//2))
            mask = np.stack([mask],axis=2)
        else:
            start_center_idx = lm_list[np.random.randint(low=0, high=len(lm_list)-1)]-1
            length = np.max([(np.max(lm[:,0])-np.min(lm[:,0]))/2, (np.max(lm[:,1])-np.min(lm[:,1]))/2])
            start_center = lm[start_center_idx,:]
            start_center[0] += length * np.random.uniform(low=-0.05, high=0.05) 
            start_center[1] += length * np.random.uniform(low=-0.05, high=0.05) 
            # scale/rotation/blur
            mask_shape = max(int(length * self.scale * 2), 10)
            mask = cv2.resize(mask,(mask_shape,mask_shape))
            mask = ndimage.rotate(mask, self.rot, reshape=False)
            mask = cv2.blur(mask, (self.blur, self.blur))
            mask = np.stack([mask],axis=2)

        self.mask = mask
        self.mask_shape = mask_shape
        self.mask_center = start_center

    def compute_mask(self, time):
        face = self.face
        mask = self.mask  
        face_shape = face.shape
        mask_shape = self.mask_shape
        mask_center = self.mask_center      

        # generate shadow mask
        movex = self.spd_x * time
        movey = self.spd_y * time
        centerx = int(mask_center[0]+movex)
        centery = int(mask_center[1]+movey)

        box = [int(centerx)-int(mask_shape/2),
               int(centery)-int(mask_shape/2),
               int(centerx)+ mask_shape - int(mask_shape/2),
               int(centery)+ mask_shape - int(mask_shape/2)]

        mbox = [0,0,mask_shape,mask_shape]
        if box[0] < 0:
            mbox[0] = -box[0]
        if box[2] > face_shape[0]:
            mbox[2] = mask_shape - (box[2]-face_shape[0])
        if box[1] < 0:
            mbox[1] = -box[1]
        if box[3] > face_shape[1]:
            mbox[3] = mask_shape - (box[3]-face_shape[1])

        box = [max(box[0],0), max(box[1],0), min(box[2],face_shape[1]), min(box[3],face_shape[0])]
        mask_to_face = np.zeros((face_shape[0],face_shape[1],face_shape[2]))
        mask_to_face[box[1]:box[3],box[0]:box[2],:] = mask[mbox[1]:mbox[3],mbox[0]:mbox[2],:]
        mask_cut = mask_to_face * self.face
        return mask_cut, self.face

    def apply_mask(self, img, img_dark):
        final_mask_ss = np.copy(self.final_mask_ss)
        final_mask_sv = np.copy(self.final_mask_sv)

        if self.incs and self.no_need:
            intensity_mask = utils.get_brightness_mask(size=(final_mask.shape[0],final_mask.shape[1]), min_val=0.5)
            final_mask_sv = final_mask_sv * np.expand_dims(intensity_mask, 2)
        shadow_face = img * final_mask_ss + img_dark * final_mask_sv
        return shadow_face

def shadow_synthesis(gt, lm, num):
    width, _, _ = gt.shape
    face = generate_face_region(lm, width)

    def face_darken(_img):
        #TONE_SIGMA = 0.2
        TONE_SIGMA = 0.3
        #TONE_SIGMA = 0.4
        #TONE_SIGMA = 0.1
        # random RGB change
        _img = _img.astype(np.float32)

        # data augmentation
        curve_gain = 0.5 + np.random.uniform(low=-TONE_SIGMA, high=TONE_SIGMA, size=(3)) 
        img_reclr = apply_tone_curve(_img, gain=curve_gain, is_rgb=True)
        color_matrix = get_ctm_ls(_img, img_reclr)
        img_aug = apply_ctm(_img, color_matrix)

        curve_gain = 0.5 + np.random.uniform(low=-TONE_SIGMA, high=TONE_SIGMA, size=(3)) 
        img_tone = apply_tone_curve(_img, gain=curve_gain, is_rgb=True)
        color_matrix = get_ctm_ls(_img, img_tone)
        img_tone = apply_ctm(_img, color_matrix)
        return img_aug, img_tone, color_matrix

    img, img_dark, color_matrix = face_darken(gt)

    # generate shadow mask
    shadow = ShadowMaker(face, lm)
    mask, face = shadow.compute_mask(num)

    return img, img_dark, mask, color_matrix, face
