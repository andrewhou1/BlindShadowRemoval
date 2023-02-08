import face_alignment
import glob
import os
import shutil
import cv2
import numpy as np
from skimage import io
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#vlist = sorted(glob.glob('./Train_files/*.avi'))
#vlist = sorted(glob.glob('./HQ/*/56766.png')) 
vlist = sorted(glob.glob('./sample_uncropped_images/*.png')) 
for vd in vlist:
    frame = cv2.imread(vd)
    #frame = cv2.resize(frame, (256,256))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(frame_rgb)
    if preds is None:
        print('No Face!')
    else: # len(preds) == 1:
        pred = preds[0] 
        lmname = vd[:-3] + 'npy'
        #lmnameb = vd[:-3] + 'npy.backup'
        #shutil.move(lmname, lmnameb)
        np.save(lmname, pred)
        #cv2.imwrite(vd, frame)
        print(lmname)
            

    #metaname = folder + '/meta'
    #face_fr = np.asarray([fr] + face_fr)
    #np.savetxt(metaname, face_fr, fmt='%d')

