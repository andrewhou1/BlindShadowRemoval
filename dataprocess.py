#import face_alignment
import glob
import os
import shutil
import cv2
import numpy as np
from skimage import io
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#vlist = sorted(glob.glob('./Train_files/*.avi'))
#vlist = sorted(glob.glob('./LQ/*.png')) 
vlist = sorted(glob.glob('./sample_uncropped_images/*.png')) 
#v = open("nvlab_list_batch_1.txt", "r")
#list = v.read().split(",")
#vlist = np.loadtxt('nvlab_list_batch_2.txt',dtype=np.int32)
for vd in vlist:
    vd_parts = vd.split('/')
    name = str(vd).split('/')[-1].split('.')[0]
    #hq = vd_parts[0]+'/HQ/'+name[:2]+'000/'+name
    hq = './sample_uncropped_images/'+name+'.png'
    hqlm = './sample_uncropped_images/'+name+'.npy'
    print(vd, hq, hqlm)
    img = cv2.imread(hq)
    pred = np.load(hqlm)
    img_shape = img.shape

    # box
    '''
    length = np.sqrt(np.sum(np.square(np.abs(pred[37, :] - pred[46, :])))) / 2 * 1.1
    eye_center = (pred[37, :] + pred[46, :]) / 2
    xl = int(eye_center[1] - length * 1.6)
    xr = int(eye_center[1] + length * 3.0)
    yl = int(eye_center[0] - length * 2.3)
    yr = int(eye_center[0] + length * 2.3)
    box = [yl,xl,yr,xr]'''

    center = [(np.min(pred[:,0])+np.max(pred[:,0]))/2, (np.min(pred[:,1])+np.max(pred[:,1]))/2]
    length = np.max([(np.max(pred[:,0])-np.min(pred[:,0]))/2, (np.max(pred[:,1])-np.min(pred[:,1]))/2]) * 1.45
    box = [int(center[0])-int(length),
           int(center[1])-int(length*1.2),
           int(center[0])+int(length),
           int(center[1])+int(length)+int(length)-int(length*1.2)]

    pred[:,0] = pred[:,0] - box[0]
    pred[:,1] = pred[:,1] - box[1]

    preset_x = 0
    preset_y = 0
    if box[0] < 0 or box[2] > img_shape[1]:
        preset_x = max(-box[0], box[2] - img_shape[1]) #int(img_shape[0]*2.5)
    if box[1] < 0 or box[3] > img_shape[0]:
        preset_y = max(-box[1], box[3] - img_shape[0]) #int(img_shape[0]*2.5)
    if preset_x > 0 or preset_y > 0:
        img_large= np.zeros((img_shape[0]+preset_y+preset_y+2,img_shape[1]+preset_x+preset_x+2,img_shape[2]))
        img_large[preset_y:preset_y+int(img_shape[0]),preset_x:preset_x+int(img_shape[1]),:] = img
        img = img_large
        box[0] = box[0] + preset_x
        box[1] = box[1] + preset_y
        box[2] = box[2] + preset_x
        box[3] = box[3] + preset_y

    face = img[box[1]:box[3],box[0]:box[2],:]

    if length > 250:
        folder = 'sample_uncropped_images_cropped/'+name
        if not os.path.isdir(folder):
           os.mkdir(folder)
        #save img
        imname = folder + '/' + name + '.png'
        lmname = folder + '/' + name + '.npy'
        face_shape = face.shape
        face = cv2.resize(face,(256,256))
        pred = pred/face_shape[0]*256
        np.save(lmname, pred)
        cv2.imwrite(imname, face, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(imname, face.shape, img.shape, length)
        

