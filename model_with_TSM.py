import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from warp import tf_batch_map_offsets
import time

class NonLocalBlock(layers.Layer):
    def __init__(self, ch=32, out_ch=None, pool=False, norm='batch'):
        super(NonLocalBlock, self).__init__()
        self.out_ch = ch if not out_ch else out_ch
        self.g     = layers.Conv2D(ch//2, (1, 1), strides=(1, 1), padding='same')
        self.phi   = layers.Conv2D(ch//2, (1, 1), strides=(1, 1), padding='same')
        self.theta = layers.Conv2D(ch//2, (1, 1), strides=(1, 1), padding='same')
        self.w     = layers.Conv2D(self.out_ch, (1, 1), strides=(1, 1), padding='same')

        self.norm = norm
        self.bnorm = layers.BatchNormalization()

        self.pool = pool
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.pool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    def call(self, x, training):
        bsize, h, w, in_ch = x.shape
        if self.pool:
            h0 = h//2
            w0 = w//2
        else:
            h0 = h + 0
            w0 = w + 0
        out_ch = self.out_ch
        # g
        g_x = self.g(x)
        if self.pool:
            g_x = self.pool1(g_x)
        g_x = tf.reshape(g_x, [bsize, h0*w0, -1]) # reshape input

        # phi
        phi_x = self.phi(x)
        if self.pool:
            phi_x = self.pool2(phi_x)
        phi_x = tf.reshape(phi_x, [bsize, h0*w0, -1]) # reshape input
        phi_x = tf.transpose(phi_x, [0,2,1])

        # theta
        theta_x = self.theta(x)
        if self.pool:
            theta_x = self.pool2(theta_x)
        theta_x = tf.reshape(theta_x, [bsize, h0*w0, -1]) # reshape input

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        y = tf.reshape(y, [bsize, h, w, -1])

        w_y = self.w(y)
        if self.norm:
            w_y = self.bnorm(w_y, training)
        z = x + w_y

        return z

class Res(layers.Layer):
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False):
        super(Res, self).__init__()
        self.conv1 = layers.Conv2D(ch, (ksize, ksize), strides=(stride, stride), padding='same')
        self.conv2 = layers.Conv2D(ch, (ksize, ksize), strides=(stride, stride), padding='same')
        self.bnorm1 = layers.BatchNormalization()
        self.bnorm2 = layers.BatchNormalization()
        self.relu1 = layers.LeakyReLU()
        self.relu2 = layers.LeakyReLU()
        self.non_local = NonLocalBlock(ch, ch)

    def call(self, x, training):
        y = self.relu1(self.conv1(self.bnorm1(x, training)))
        y = self.conv2(self.bnorm2(y, training))
        y = self.relu2(x+y)
        y = self.non_local(y)
        return y

class ResBottleneck(layers.Layer):
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False):
        super(ResBottleneck, self).__init__()
        self.conv1 = layers.Conv2D(ch//2, (1, 1), strides=(1, 1), padding='same')
        self.conv2 = layers.Conv2D(ch//2, (ksize, ksize), strides=(stride, stride), padding='same')
        self.conv3 = layers.Conv2D(ch, (1, 1), strides=(1, 1), padding='same')
        self.bnorm1 = layers.BatchNormalization()
        self.bnorm2 = layers.BatchNormalization()
        self.bnorm3 = layers.BatchNormalization()
        self.relu1 = layers.LeakyReLU()
        self.relu2 = layers.LeakyReLU()
        self.relu3 = layers.LeakyReLU()
        self.stride = stride
        self.non_local = NonLocalBlock(ch, ch)
        if stride > 1:
            self.conv_red = layers.Conv2D(ch, (1, 1), strides=(stride, stride), padding='same')

    def call(self, x, training):
        y = self.relu1(self.bnorm1(self.conv1(x), training))
        y = self.relu2(self.bnorm2(self.conv2(y), training))
        y = self.bnorm3(self.conv3(y), training)
        y = self.non_local(y)
        if self.stride > 1:
            x = self.conv_red(x)
        if x.shape[-1] < y.shape[-1]:
            ch_add =  y.shape[-1] - x.shape[-1]
            ch_pad = tf.zeros([x.shape[0],x.shape[1],x.shape[2],ch_add])
            x = tf.concat([x,ch_pad],axis=3)
        elif y.shape[-1] < x.shape[-1]:
            ch_add =  x.shape[-1] - y.shape[-1]
            ch_pad = tf.zeros([y.shape[0],y.shape[1],y.shape[2],ch_add])
            y = tf.concat([y,ch_pad],axis=3)
        return self.relu3(x+y)

class Conv(layers.Layer):
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False, name=None):
        super(Conv, self).__init__()
        self.norm = norm
        self.conv = layers.Conv2D(ch, (ksize, ksize), strides=(stride, stride), padding='same',name=name)
        if norm == 'batch':
            # conv + bn
            self.bnorm = layers.BatchNormalization()
        else:
            self.bnorm = None
        if norm == 'spec':
            # conv + sn
            self.conv = tfa.layers.SpectralNormalization(self.conv)
        # relu
        if nl:
            self.relu = layers.LeakyReLU()
        else:
            self.relu = None
        # dropout
        if dropout:
            self.drop = layers.Dropout(0.3)
        else:
            self.drop = None

    def call(self, x, training):
        x = self.conv(x)
        if self.bnorm:
            x = self.bnorm(x, training)
        if self.relu:
            x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x

class ConvT(layers.Layer):
    def __init__(self, ch=32, ksize=3, stride=2, norm='batch', nl=True, dropout=False):
        super(ConvT, self).__init__()
        self.norm = norm
        self.conv = layers.Conv2DTranspose(ch, (ksize, ksize), strides=(stride, stride), padding='same')
        if norm == 'batch': # conv + bn
            self.bnorm = layers.BatchNormalization()
        else:
            self.bnorm = None
        if norm == 'spec': # conv + sn
            self.conv = tfa.layers.SpectralNormalization(self.conv)
        if nl: # relu
            self.relu = layers.LeakyReLU()
        else:
            self.relu = None
        if dropout: # dropout
            self.drop = layers.Dropout(0.3)
        else:
            self.drop = None

    def call(self, x, training):
        x = self.conv(x)
        if self.bnorm:
            x = self.bnorm(x, training)
        if self.relu:
            x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x

'''class ShareLayer(layers.Layer):
    def __init__(self):
        super(ShareLayer, self).__init__(self)
        self.imsize = 256

    def call(self, x, reg, chuck):
        reg_in, reg_out = tf.split(reg, 2, axis=3)
        x_reg = tf_batch_map_offsets(x, reg_in)
        # reshape
        chuck_bsize,w,h,ch = x_reg.shape
        x_reg = tf.reshape(x_reg,[chuck_bsize//chuck,chuck,w,h,ch])
        x_max = tf.reduce_max(x_reg, axis=1)
        x_mean = tf.reduce_mean(x_reg, axis=1)
        x_share = tf.concat([x_max, x_mean], axis=3)
        x_share = tf.stack([x_share for _ in range(chuck)], axis=1)
        x_share = tf.reshape(x_share, [chuck_bsize,w,h,-1])
        x_share_dereg = tf_batch_map_offsets(x_share, reg_out)
        return x_share_dereg'''

class ShareLayer(layers.Layer):
    def __init__(self):
        super(ShareLayer, self).__init__(self)
        self.imsize = 256

    def call(self, x, reg, frame, share):
        #print(tf.shape(x))
        #time.sleep(100)
        reg_in, reg_out = tf.split(reg, 2, axis=3)
        x_reg = tf_batch_map_offsets(x, reg_in)
        # reshape
        chuck_bsize,w,h,ch = x_reg.shape
        print(chuck_bsize)
        print(w)
        print(h)
        print(ch)
        print(frame)
        #time.sleep(100)
        #x_reg_1 = tf.reshape(x_reg,[chuck_bsize//frame//2,frame*2,w,h,ch])
        x_reg_1 = tf.reshape(x_reg,[1,frame,w,h,ch])
        x_max_1 = tf.reduce_max(x_reg_1, axis=1)
        x_mean_1 = tf.reduce_mean(x_reg_1, axis=1)
        x_share_1 = tf.concat([x_max_1, x_mean_1], axis=3)
        #x_share_1 = tf.stack([x_share_1 for _ in range(frame*2)], axis=1)
        x_share_1 = tf.stack([x_share_1 for _ in range(frame)], axis=1)
        x_share_1 = tf.reshape(x_share_1, [chuck_bsize,w,h,-1])
        x_share_1 = tf_batch_map_offsets(x_share_1, reg_out)

        x_share_2 = tf.concat([x, x], axis=3)
        x_share = tf.cond(share, lambda: x_share_1, lambda: x_share_2)
        return x_share

class Generator(tf.keras.Model):
    def __init__(self, downsize=1, n_res=6):
        super(Generator, self).__init__()
        n_ch = [32,64,64,96,128,256,256]
        self.n_ch = n_ch
        self.conv1 = Conv(n_ch[0], ksize=7, name='tconv1')
        self.conv2 = Conv(1, ksize=7, norm=False, nl=False, name='tconv3')
        self.conv3 = Conv(1, ksize=7, norm=False, nl=False, name='tconv4')

        self.down1 = Conv(n_ch[1], stride=2)
        self.down2 = Conv(n_ch[2], stride=2)
        self.down3 = Conv(n_ch[3], stride=2)
        self.up1 = ConvT(n_ch[3])
        self.up2 = ConvT(n_ch[2])
        self.up3 = ConvT(n_ch[1])

        self.clr_up1 = ConvT(n_ch[4])
        self.clr_up2 = ConvT(n_ch[3])
        self.clr_up3 = ConvT(n_ch[2])
        self.clr_conv1 = Conv(16, ksize=3)
        self.clr_conv2 = Conv(16, ksize=1)
        self.clr_conv3 = Conv(3, ksize=1, norm=False, nl=False)

        self.info_share = ShareLayer()

        self.n_res = n_res
        self.res_stack = []
        for i in range(n_res):
            self.res_stack.append(ResBottleneck(n_ch[5]+1, ksize=3, stride=1, norm='batch'))

    def call(self, inputs, uv, reg, frame, share, chuck, training):
        # header
        x1 = self.conv1(inputs, training)
        x2 = self.down1(x1, training)
        x3 = self.down2(x2, training)
        x  = self.down3(x3, training)
        b,w,h,ch = x.shape

        # information sharing
        uv = tf.image.resize(uv, [w,h])
        x_share = self.info_share(x, reg, frame, share)
        x = tf.concat([x, x_share, uv],axis=3)
        for i in range(self.n_res//2):
            x = self.res_stack[i](x, training)

        # greyscale
        y = self.up1(x, training)
        y = self.up2(tf.concat([y,x3],axis=3), training)
        y = self.up3(tf.concat([y,x2],axis=3), training)
        mask = tf.tanh(self.conv2(y, training))
        con = self.conv3(y, training)
        #gs = tf.image.rgb_to_grayscale(tf.reverse(inputs, axis=[-1]))*(1+mask)+con
        #dif = gs - tf.image.rgb_to_grayscale(tf.reverse(inputs, axis=[-1]))
        gs = tf.image.rgb_to_grayscale(inputs)*(1+mask)+con
        dif = gs - tf.image.rgb_to_grayscale(inputs)
        mask22 = tf.concat([tf.nn.relu(mask),mask*0,tf.nn.relu(-mask)],axis=3)

        # rgb
        #bmask = tf.cast(tf.greater(tf.stop_gradient(tf.image.resize(dif, [w,h])), 0.12),tf.float32)
        bmask = tf.cast(tf.greater(tf.stop_gradient(tf.image.resize(dif, [w,h])), 0.1),tf.float32)
        #bmask = tf.cast(tf.greater(tf.stop_gradient(tf.image.resize(dif, [w,h])), 0.08),tf.float32)
        x_hole = x*(1-bmask)
        x_share = self.info_share(x_hole, reg, frame, share)
        x = tf.concat([x_hole, bmask, x_share, uv], axis=3)
        bmask_progress = []
        for i in range(self.n_res//2,self.n_res):
            x = self.res_stack[i](x, training)

        f = self.clr_up1(x, training) #32
        f = self.clr_up2(f, training) #16
        f = self.clr_up3(f, training) #16
        con_rgb = self.clr_conv1(tf.concat([gs,f],axis=3), training)
        con_rgb = self.clr_conv2(con_rgb, training)
        con_rgb = self.clr_conv3(con_rgb, training)

        '''
        reg_in, reg_out = tf.split(reg, 2, axis=3)
        x_reg = tf_batch_map_offsets(con_rgb, reg_in)
        chuck_bsize,w,h,ch = x_reg.shape
        x_reg = tf.reshape(x_reg,[chuck_bsize//10,10,w,h,ch])
        x_max = tf.reduce_mean(x_reg, axis=1)
        x_share = tf.stack([x_max for _ in range(10)], axis=1)
        x_share = tf.reshape(x_share, [chuck_bsize,w,h,-1])
        con_rgb_share = tf_batch_map_offsets(x_share, reg_out)
        bmask  = tf.cast(tf.greater(tf.stop_gradient(tf.image.resize(dif, [32,32])), 0.08),tf.float32)
        bmask  = tfa.image.gaussian_filter2d(tf.image.resize(bmask, [256,256], method='gaussian'), [7,7])
        #kernel = tf.ones((3,3,1))
        #bmask = tf.nn.dilation2d(bmask, kernel, [1,1,1,1], 'SAME', 'NHWC', [1,1,1,1]) 
        #bmask /= tf.reduce_max(bmask)
        con_rgb = con_rgb * (1-bmask) + con_rgb_share * bmask
        '''

        dif = tf.image.rgb_to_grayscale(con_rgb) - tf.image.rgb_to_grayscale(inputs)

        return gs, con_rgb, mask22, dif #tf.cast(tf.greater(dif, 0.04),tf.float32)

class Discriminator(tf.keras.Model):
    def __init__(self, downsize=1, num_layers=3):
        super(Discriminator, self).__init__()
        n_ch = [32,32,64,64,128,256]
        #self.conv1 = Conv(n_ch[0], ksize=4, stride=2, norm=False)
        self.conv_stack = []
        for i in range(num_layers):
            self.conv_stack.append(Conv(n_ch[i], ksize=4, stride=2, norm='batch'))
        self.conv2 = Conv(1, ksize=4, norm=False, nl=False)
        self.downsize = downsize
        self.num_layers = num_layers

    def call(self, x, training):
        if self.downsize > 1:
            _,w,h,_ = x.shape 
            x=tf.image.resize(x,(w//self.downsize,h//self.downsize))
        #x = self.conv1(x, training)
        for i in range(self.num_layers):
            x = self.conv_stack[i](x, training)
        x = self.conv2(x)
        return tf.split(x,2,axis=0)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
