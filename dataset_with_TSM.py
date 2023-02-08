import cv2
import tensorflow as tf
import glob
import random
import numpy as np
import time
from natsort import natsorted, ns
from utils import generate_landmark_map, face_crop_and_resize, shadow_synthesis, generate_face_region, generate_face_region2
from warp import generate_offset_map, generate_uv_map
autotune = tf.data.experimental.AUTOTUNE
uv=[[0.19029412,0.19795537 ,0.21318457 ,0.22828290 ,0.24970947 ,0.28816611 ,0.33394283 ,0.39239809 ,0.47876307 ,0.56515092 ,0.62323409 ,0.66867208 ,0.70676976 ,0.72820741 ,0.74272829 ,0.75663871 ,0.76398379 ,0.25338903 ,0.28589997 ,0.32738855 ,0.36722445 ,0.40321609 ,0.55088127 ,0.58705842 ,0.62712812 ,0.66933709 ,0.70184904 ,0.47813031 ,0.47830373 ,0.47872066 ,0.47870359 ,0.43102017 ,0.45095450 ,0.47804111 ,0.50489837 ,0.52461874 ,0.30827355 ,0.33330417 ,0.36890128 ,0.40203944 ,0.37214473 ,0.33496466 ,0.55122417 ,0.58458656 ,0.62106317 ,0.64688802 ,0.61956245 ,0.58191341 ,0.37796655 ,0.41338006 ,0.45562238 ,0.47811818 ,0.50052267 ,0.54254669 ,0.57570505 ,0.54044306 ,0.51024377 ,0.47821599 ,0.44642609 ,0.41657540 ,0.38790068 ,0.44901687 ,0.47766650 ,0.50653827 ,0.56918079 ,0.50583494 ,0.47757983 ,0.44971457],
    [0.55190903,0.47428983 ,0.40360034 ,0.33980367 ,0.27118790 ,0.21624640 ,0.18327993 ,0.15577883 ,0.14014046 ,0.15676366 ,0.18313733 ,0.21531384 ,0.26951864 ,0.33780637 ,0.40212137 ,0.47324431 ,0.55168754 ,0.63735390 ,0.66241443 ,0.67068136 ,0.66713846 ,0.65712863 ,0.65805173 ,0.66828096 ,0.67205220 ,0.66368717 ,0.63796753 ,0.58252430 ,0.53523010 ,0.48812559 ,0.44775373 ,0.41256407 ,0.40846801 ,0.40317070 ,0.40854913 ,0.41281027 ,0.58095986 ,0.59604895 ,0.59652811 ,0.57966459 ,0.57139677 ,0.56953919 ,0.57967824 ,0.59695679 ,0.59599525 ,0.58050835 ,0.57008123 ,0.57134289 ,0.31730300 ,0.34064898 ,0.35593933 ,0.35154018 ,0.35593045 ,0.34062389 ,0.31715956 ,0.30086508 ,0.28950119 ,0.28752795 ,0.28963783 ,0.30076182 ,0.31932616 ,0.32959232 ,0.33032984 ,0.32936266 ,0.31900606 ,0.32014942 ,0.31873652 ,0.32043788],
    [0.54887491,0.55835652 ,0.56531715 ,0.58029217 ,0.61638439 ,0.68007606 ,0.75769442 ,0.82921398 ,0.85709274 ,0.82894272 ,0.75751764 ,0.68032110 ,0.61664295 ,0.58068472 ,0.56520522 ,0.55785143 ,0.54947090 ,0.79504120 ,0.84203368 ,0.87477297 ,0.89484525 ,0.90437353 ,0.90412331 ,0.89423305 ,0.87385195 ,0.84139013 ,0.79445726 ,0.91648984 ,0.95176858 ,0.98838627 ,0.99706292 ,0.91018295 ,0.92791700 ,0.93613458 ,0.92778808 ,0.90999144 ,0.82165444 ,0.85368645 ,0.85440493 ,0.84463143 ,0.85324180 ,0.84432119 ,0.84337026 ,0.85280263 ,0.85272932 ,0.82140154 ,0.84402239 ,0.85248041 ,0.86857969 ,0.91266698 ,0.93638903 ,0.93873996 ,0.93629760 ,0.91227442 ,0.86774820 ,0.90530455 ,0.92216164 ,0.92610627 ,0.92281538 ,0.90596151 ,0.87151438 ,0.91635096 ,0.92336667 ,0.91626322 ,0.87006092 ,0.91713434 ,0.92056626 ,0.91682398]]
uv = np.transpose(np.asarray(uv, dtype=np.float32))
lm_ref = [[42.022587,44.278061,48.761536,53.206482,59.514465,70.836105,84.312767,101.52200,126.94785,152.38043,169.48012,182.85706,194.07301,200.38426,204.65921,208.75444,210.91682,60.597733,70.168953,82.383194,94.110878,104.70682,148.17944,158.83000,170.62653,183.05284,192.62436,126.76157,126.81262,126.93536,126.93034,112.89234,118.76100,126.73531,134.64207,140.44775,76.755737,84.124748,94.604538,104.36041,95.559410,84.613594,148.28040,158.10228,168.84100,176.44383,168.39919,157.31531,97.273354,107.69909,120.13522,126.75800,133.35388,145.72574,155.48756,145.10645,136.21576,126.78679,117.42784,108.63980,100.19796,118.19057,126.62502,135.12486,153.56682,134.91780,126.59950,118.39597],
          [94.517975,117.36908,138.18005,156.96179,177.16229,193.33707,203.04239,211.13872,215.74265,210.84879,203.08437,193.61160,177.65372,157.54980,138.61548,117.67688,94.583191,69.363007,61.985199,59.551407,60.594437,63.541336,63.269577,60.258087,59.147827,61.610504,69.182358,85.504852,99.428253,113.29582,125.18130,135.54114,136.74701,138.30655,136.72314,135.46866,85.965424,81.523193,81.382126,86.346741,88.780792,89.327667,86.342728,81.255920,81.539001,86.098343,89.168091,88.796661,163.58600,156.71295,152.21146,153.50656,152.21408,156.72034,163.62823,168.42532,171.77084,172.35178,171.73062,168.45572,162.99039,159.96802,159.75090,160.03563,163.08463,162.74802,163.16397,162.66309]]
lm_ref = np.transpose(np.asarray(lm_ref, dtype=np.float32))/256.

class Dataset():
    def __init__(self, config, mode, dset=None):
        self.config = config
        self.mode = mode
        self.dset = dset
        if mode == 'train':
            data_dir = config.DATA_DIR
        elif mode == 'val':
            data_dir = config.DATA_DIR_VAL
        else:
            data_dir = config.DATA_DIR_TEST
        self.input_tensors, self.name_list = self.inputs(data_dir)
        self.feed = iter(self.input_tensors)

    def inputs(self, data_dir):
        mode =  self.mode
        if mode == 'train' or mode == 'val':
            #print("Here1")
            #time.sleep(100)
            data_samples = []
            for _dir in data_dir:
                _list = glob.glob(_dir)
                data_samples += _list
            shuffle_buffer_size = len(data_samples)
            dataset = tf.data.Dataset.from_tensor_slices(data_samples)
            dataset = dataset.cache()
            dataset = dataset.shuffle(shuffle_buffer_size).repeat(-1)
            dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        else:
            #print("Here2")
            #time.sleep(100)
            data_samples = []
            vid_count = 0
            for _dir in data_dir:
                for _file in natsorted(glob.glob(_dir)):
                    #print(_dir)
                    vid_count += 1
                    #print(vid_count)
                    #if(vid_count < 82):
                    #    continue

                    #_list = natsorted(glob.glob(_file+'/*.npy'))
                    _list = natsorted(glob.glob(_file+'/*_label.png'))
                    #data_samples += _list[:90]
                    data_samples += _list
            shuffle_buffer_size = len(data_samples)
            dataset = tf.data.Dataset.from_tensor_slices(data_samples)
            dataset = dataset.cache()
            if self.dset == 'sfw':
                #print("Here1")
                #time.sleep(100)
                dataset = dataset.map(map_func=self.parse_fn_test_sfw, num_parallel_calls=autotune)
                #dataset = dataset.map(map_func=self.parse_fn_test_sfw_video, num_parallel_calls=autotune)
                dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
            else:
                #print("Here2")
                #time.sleep(100)
                dataset = dataset.map(map_func=self.parse_fn_test, num_parallel_calls=autotune)
                dataset = dataset.batch(batch_size=1).prefetch(buffer_size=autotune)
        return dataset, data_samples

    def parse_fn(self, file):
        config = self.config
        def _parse_function(_path):
            _path = _path.decode('UTF-8')
            _list = glob.glob(_path+'/*.npy')
            _file = _list[random.randint(0, len(_list) - 1)]
            _gt   = _file.split('.')[0]+'.png'
            _lm   = _file

            gt = cv2.cvtColor(cv2.imread(_gt), cv2.COLOR_BGR2RGB) / 255.
            
            gt, lm, lm_mirror, _= face_crop_and_resize(gt, np.load(_lm), config.IMG_SIZE, aug=True)
            gt, img_dark, mask, color_matrix, face  = shadow_synthesis(gt, lm, 0)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img = np.concatenate([img_dark, gt, mask, uvm, reg_in, reg_out, face], axis=2)

            # img2
            img_dark_m = cv2.flip(img_dark, 1)
            gt_m = cv2.flip(gt, 1)
            face_m = cv2.flip(face, 1).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            mask_m = cv2.flip(mask, 1).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            uvm_mirror = generate_uv_map(lm_mirror, uv, config.IMG_SIZE)
            reg_in_m = generate_offset_map(lm_mirror, lm_ref, config.IMG_SIZE)
            reg_out_m= generate_offset_map(lm_ref, lm_mirror, config.IMG_SIZE)
            img2 = np.concatenate([img_dark_m, gt_m, mask_m, uvm_mirror, reg_in_m, reg_out_m, face_m], axis=2)

            img_chuck = np.stack([img,img2],axis=0)
            return img_chuck.astype(np.float32), _gt

        _img, _name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.string])
        _img = tf.ensure_shape(_img, [2, config.IMG_SIZE, config.IMG_SIZE, 17])
        return _img, _name
    '''
    def parse_fn_test(self, file):
        config = self.config
        def _parse_function(_lm):
            _lm = _lm.decode('UTF-8')
            _lm_part = _lm.split('/')
            _img = _lm.split('.')[0]+'.png'
            _gt  = '/'.join(_lm_part[0:6]+['gt']+_lm_part[-2:]).split('.')[0]+'.png'

            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            gt  = cv2.cvtColor(cv2.imread(_gt), cv2.COLOR_BGR2RGB) / 255.
            img = np.concatenate([img,gt],axis=2)

            sz0 = img.shape[0]
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img1 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2) # 3,3,1,3,3,1

            # img2
            img_m = cv2.flip(img, 1)
            uvm_mirror = generate_uv_map(lm_mirror, uv, config.IMG_SIZE)
            reg_in_m = generate_offset_map(lm_mirror, lm_ref, config.IMG_SIZE)
            reg_out_m= generate_offset_map(lm_ref, lm_mirror, config.IMG_SIZE)
            face_m = generate_face_region(lm_mirror, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            img2 = np.concatenate([img_m, uvm_mirror, reg_in_m, reg_out_m, face_m], axis=2)

            img_chuck = np.stack([img1,img2],axis=0)
            return img_chuck.astype(np.float32), np.asarray(box,np.float32), _gt

        _img, _box, _name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.float32, tf.string])
        _img = tf.ensure_shape(_img, [2, config.IMG_SIZE, config.IMG_SIZE, 16])
        _box = tf.ensure_shape(_box, [4])
        return _img, _box, _name
        '''

    def parse_fn_test(self, file):
        config = self.config
        def _parse_function(_lm):
            _lm = _lm.decode('UTF-8')
            _lm_part = _lm.split('/')
            _img = _lm.split('.')[0]+'.png'
            _gt  = '/'.join(_lm_part[0:7]+['gt']+_lm_part[-2:]).split('.')[0]+'.png'
            print(_img)
            print(_gt)

            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            gt  = cv2.cvtColor(cv2.imread(_gt), cv2.COLOR_BGR2RGB) / 255.
            img = np.concatenate([img,gt],axis=2)

            sz0 = img.shape[0]
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img1 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2) # 3,3,1,3,3,1

            # img2
            img_m = cv2.flip(img, 1)
            uvm_mirror = generate_uv_map(lm_mirror, uv, config.IMG_SIZE)
            reg_in_m = generate_offset_map(lm_mirror, lm_ref, config.IMG_SIZE)
            reg_out_m= generate_offset_map(lm_ref, lm_mirror, config.IMG_SIZE)
            face_m = generate_face_region(lm_mirror, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            img2 = np.concatenate([img_m, uvm_mirror, reg_in_m, reg_out_m, face_m], axis=2)

            img_chuck = np.stack([img1,img2],axis=0)
            return img_chuck.astype(np.float32), np.asarray(box,np.float32), _gt

        _img, _box, _name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.float32, tf.string])
        _img = tf.ensure_shape(_img, [2, config.IMG_SIZE, config.IMG_SIZE, 16])
        _box = tf.ensure_shape(_box, [4])
        return _img, _box, _name

    '''
    def parse_fn_test_sfw(self, file):
        config = self.config
        def _parse_function(_lm):
            _lm = _lm.decode('UTF-8')
            _lm_part = _lm.split('/')
            _img = _lm.split('.')[0]+'.png'

            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.

            sz0 = img.shape[0]
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img1 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            # img2
            img_m = cv2.flip(img, 1)
            uvm_mirror = generate_uv_map(lm_mirror, uv, config.IMG_SIZE)
            reg_in_m = generate_offset_map(lm_mirror, lm_ref, config.IMG_SIZE)
            reg_out_m= generate_offset_map(lm_ref, lm_mirror, config.IMG_SIZE)
            face_m = generate_face_region(lm_mirror, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            img2 = np.concatenate([img_m, uvm_mirror, reg_in_m, reg_out_m, face_m], axis=2)

            img_chuck = np.stack([img1,img2],axis=0)
            return img_chuck.astype(np.float32), np.asarray(box, np.float32), _img

        _img, _box, _name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.float32, tf.string])
        _img = tf.ensure_shape(_img, [2, config.IMG_SIZE, config.IMG_SIZE, 13])
        _box = tf.ensure_shape(_box, [4])
        return _img, _box, _name
        '''
    def parse_fn_test_sfw(self, file):
        config = self.config
        def _parse_function(_mask):
            _mask = _mask.decode('UTF-8')
            _lm = _mask.split('.')[0]
            _lm = _lm[:-6]+'.npy'
            _cmap = _mask.split('.')[0]+'_cmap.png'
            _lm_part = _lm.split('/')
            _img = _lm.split('.')[0]+'.png'
            frame = int(_lm_part[-1].split('.')[0])
            print(_img)
            print(_cmap)
            print(_mask)
            #time.sleep(100)

            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            cmap = cv2.cvtColor(cv2.imread(_cmap), cv2.COLOR_BGR2RGB) / 255.
            mask = cv2.imread(_mask, 0)
            mask = np.stack([mask],axis=2)
            print(np.shape(img))
            print(np.shape(cmap))
            print(np.shape(mask))
            #time.sleep(100)
            #img = np.reshape(cv2.resize(img, (256, 256)), (256, 256, 3))
            #cmap = np.reshape(cv2.resize(cmap, (256, 256)), (256, 256, 3))
            #mask = np.reshape(cv2.resize(mask, (256, 256)), (256, 256, 1))
            img = np.concatenate([img, cmap, mask],axis=2)
            #print("Here")

            sz0 = img.shape[0]
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img1 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)
            #print("Here 2")

            '''img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            img_m = cv2.flip(img, 1)
            cmap_m = cv2.flip(cmap, 1)
            mask = cv2.imread(_mask, 0)
            mask_m = cv2.flip(mask, 1)
            mask_m = np.stack([mask_m], axis=2)'''
            #img_m = np.reshape(cv2.resize(img_m, (256, 256)), (256, 256, 3))
            #cmap_m = np.reshape(cv2.resize(cmap_m, (256, 256)), (256, 256, 3))
            #mask_m = np.reshape(cv2.resize(mask_m, (256, 256)), (256, 256, 1))
            #img_m = np.concatenate([img_m, cmap_m, mask_m], axis=2)
            img_m = cv2.flip(img1[:, :, :7], 1)
            uvm_mirror = generate_uv_map(lm_mirror, uv, config.IMG_SIZE)
            reg_in_m = generate_offset_map(lm_mirror, lm_ref, config.IMG_SIZE)
            reg_out_m= generate_offset_map(lm_ref, lm_mirror, config.IMG_SIZE)
            face_m = generate_face_region(lm_mirror, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            img2 = np.concatenate([img_m, uvm_mirror, reg_in_m, reg_out_m, face_m], axis=2)
            #print(np.shape(img1))
            #print(np.shape(img2))
            img_chuck = np.stack([img1,img2],axis=0)
            return img_chuck.astype(np.float32), np.asarray(box, np.float32), _img

        _img, _box, _name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.float32, tf.string])
        _img = tf.ensure_shape(_img, [2, config.IMG_SIZE, config.IMG_SIZE, 17])
        _box = tf.ensure_shape(_box, [4])
        return _img, _box, _name

    def parse_fn_test_sfw_video(self, file):
        config = self.config
        def _parse_function(_mask):
            _mask = _mask.decode('UTF-8')
            _lm = _mask.split('.')[0]+'.npy'
            #_lm = _lm[:-6]+'.npy'
            _cmap = _mask.split('.')[0]+'_cmap.png'
            _lm_part = _lm.split('/')
            #_img = _lm.split('.')[0]+'.png'
            _img = _mask.split('.')[0]+'.png'
            frame = int(_lm_part[-1].split('.')[0])

            print(_img)
            print(_cmap)
            #print("Mask: "+_mask)
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            #cmap = cv2.cvtColor(cv2.imread(_cmap), cv2.COLOR_BGR2RGB) / 255.
            #mask = cv2.imread(_mask, 0)
            #print("Read in Mask")
            #print(np.shape(mask))
            #print(np.shape([mask]))
            #mask = np.stack([mask],axis=2)
            #print("Stacked mask")
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            #print("Concatenated")

            sz0 = img.shape[0]
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img1 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            # img2
            if frame < 3:
                frame2 = frame + 2
                frame3 = frame + 4
                frame4 = frame + 6
                frame5 = frame + 8
                frame6 = frame + 10
                frame7 = frame + 12
                frame8 = frame + 14
                frame9 = frame + 16
                frame10 = frame + 1 
            elif frame <5:
                frame2 = frame + 1
                frame3 = frame + 3
                frame4 = frame + 5
                frame5 = frame + 7
                frame6 = frame + 9
                frame7 = frame + 11
                frame8 = frame + 13
                frame9 = frame + 15
                frame10 = frame - 2 
            elif frame <7:
                frame2 = frame + 1
                frame3 = frame + 3
                frame4 = frame + 5
                frame5 = frame + 7
                frame6 = frame + 9
                frame7 = frame + 11
                frame8 = frame + 13
                frame9 = frame - 2
                frame10 = frame - 4 
            elif frame <9:
                frame2 = frame + 1
                frame3 = frame + 3
                frame4 = frame + 5
                frame5 = frame + 7
                frame6 = frame + 9
                frame7 = frame + 11
                frame8 = frame - 2
                frame9 = frame - 4
                frame10 = frame - 6 
            elif frame >100:
                frame2 = frame - 1
                frame3 = frame - 3
                frame4 = frame - 5
                frame5 = frame - 7
                frame6 = frame - 9
                frame7 = frame - 11
                frame8 = frame - 2
                frame9 = frame - 4
                frame10 = frame - 6 
            else:
                frame2 = frame + 1
                frame3 = frame + 3
                frame4 = frame + 5
                frame5 = frame + 7
                frame6 = frame + 9
                frame7 = frame - 2
                frame8 = frame - 4
                frame9 = frame - 6
                frame10 = frame - 8 

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame2)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame2)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 2'+_mask)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img2 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame3)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame3)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 3'+_mask)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img3 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame4)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame4)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 4'+_mask)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img4 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame5)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame5)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 5'+_mask)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img5 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame6)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame6)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 6'+_mask)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img6 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame7)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame7)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 7'+_mask)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img7 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame8)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame8)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 8'+_mask)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img8 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame9)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame9)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 9'+_mask+_img)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img9 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            _img = '/'.join(_lm_part[:-1])+'/'+str(frame10)+'.png'
            _lm = '/'.join(_lm_part[:-1])+'/'+str(frame10)+'.npy'
            if cv2.imread(_img) is None:
                print(frame)
                print('frame 10'+_mask+_img)
                input()
            img = cv2.cvtColor(cv2.imread(_img), cv2.COLOR_BGR2RGB) / 255.
            sz0 = img.shape[0]
            sz1 = img.shape[1]
            #cmap = cv2.resize(cmap, (sz1,sz0))
            #mask = cv2.resize(mask, (sz1,sz0))
            #mask = np.stack([mask],axis=2)
            #img = np.concatenate([img, cmap, mask],axis=2)
            #img = np.concatenate([img, mask],axis=2)
            img, lm, lm_mirror, box = face_crop_and_resize(img, np.load(_lm), config.IMG_SIZE)
            uvm = generate_uv_map(lm, uv, config.IMG_SIZE)
            face = generate_face_region(lm, 256).reshape(config.IMG_SIZE,config.IMG_SIZE,1)
            reg_in = generate_offset_map(lm, lm_ref, config.IMG_SIZE)
            reg_out= generate_offset_map(lm_ref, lm, config.IMG_SIZE)
            img10 = np.concatenate([img, uvm, reg_in, reg_out, face], axis=2)

            img_chuck = np.stack([img1,img2,img3,img4,img5,img6,img7,img8,img9,img10],axis=0)
            return img_chuck.astype(np.float32), np.asarray(box, np.float32), _img

        _img, _box, _name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.float32, tf.string])
        #_img = tf.ensure_shape(_img, [10, config.IMG_SIZE, config.IMG_SIZE, 17])
        _img = tf.ensure_shape(_img, [10, config.IMG_SIZE, config.IMG_SIZE, 13])
        _box = tf.ensure_shape(_box, [4])
        return _img, _box, _name



