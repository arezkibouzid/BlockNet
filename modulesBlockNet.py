import tensorflow as tf
tf.config.run_functions_eagerly(True)
        
# FeaturePyramidExtractor ----------------------------------
class FeaturePyramidExtractor(tf.keras.Sequential):
    """ Feature pyramid extractor module"""
    def __init__(self, num_levels,filters, name = 'fp_extractor'):
        super().__init__()
        assert len(filters) == num_levels
        self.num_levels = num_levels
        #self.filters = [16, 32, 64]
        self.filters = filters
        #self.name = name
 
    def __call__(self, images):
        """
        Args:
        - images (batch, h, w, 3): input images
 
        Returns:
        - features_pyramid (batch, h_l, w_l, nch_l) for each scale levels:
          extracted feature pyramid (deep -> shallow order)
        """
        features_pyramid = []
        x = images
        for l in range(self.num_levels):
            x = tf.keras.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
            x= tf.nn.leaky_relu(x,0.1)
            x = tf.keras.layers.Conv2D(self.filters[l], (3, 3),dilation_rate=(2, 2),padding='same')(x)
            x= tf.nn.leaky_relu(x,0.1)              
            features_pyramid.append(x)
            # return feature pyramid by ascent order
        return features_pyramid[::-1]
        

# Warping layer ---------------------------------
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)
            
    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11

class WarpingLayer(tf.keras.layers.Layer):
    def __init__(self, warp_type = 'bilinear', name = 'Warping_layer'):
        super().__init__()
        self.warp = warp_type
        #self.name = name
        assert self.warp in ['nearest', 'bilinear']


    def __call__(self, x, flow):
        # expect shape
        # x:(#batch, height, width, #channel)
        # flow:(#batch, height, width, 2)
        if self.warp == 'nearest':
            x_warped = nearest_warp(x, flow)
        else:
            x_warped = bilinear_warp(x, flow)
        
        return x_warped


# Cost volume layer -------------------------------------
def extract_patch(f_c_r,s_range,block_size=-1):
    if(block_size==-1):
        block_size=s_range
    liste_images = tf.unstack(f_c_r,axis=0)
    blist =[]
    for image in liste_images:
        blist.append(tf.unstack(image,axis=2))
    result =[]
    image_block =[]
    for image in blist:
        for fetMap in image:
            fetMap= tf.reshape(fetMap,[1,fetMap.shape[0],fetMap.shape[1],1])
            pp = tf.image.extract_patches(images=fetMap,sizes=[1, s_range, s_range, 1],strides=[1, block_size, block_size, 1],rates=[1, 1, 1, 1],padding='SAME')
            pp = tf.squeeze(pp)
            pp = tf.reshape(pp,shape=(pp.shape[0],pp.shape[1],s_range,s_range))
            pp = tf.transpose(pp,[2,3,0,1])
            pp = tf.reshape(pp,(pp.shape[0],pp.shape[1],pp.shape[2]*pp.shape[3]))
            image_block.append(pp) #stack 
        tensor_image = tf.convert_to_tensor(image_block, dtype=tf.float32)
        result.append(tensor_image)
        image_block.clear()
    result = tf.convert_to_tensor(result, dtype=tf.float32)
    return result


class CostVolumeLayer(tf.keras.layers.Layer):
    """ Cost volume module """
    def __init__(self, search_range = 15,block_size = 4, name = 'Cost_Volume_layer'):
        super().__init__()
        self.s_range = search_range
        self.block_size = block_size 
        #self.name = name
 
    def __call__(self,features_0, features_0from1):
        with tf.name_scope(self.name) as ns:
            f_c=features_0
            shape = f_c.shape
            f_r=features_0from1

            f_c=tf.keras.layers.AveragePooling2D(pool_size=(self.block_size, self.block_size), strides=(self.block_size,self.block_size))(f_c)            
            f_r=tf.keras.layers.AveragePooling2D(pool_size=(self.block_size, self.block_size), strides=(1,1))(f_r)

            f_c_r=tf.repeat(tf.repeat(f_c, self.s_range, axis=1), self.s_range, axis=2)
            
            p_c = extract_patch(f_c_r,self.s_range)
            p_r = extract_patch(f_r,self.s_range,self.block_size)
            cv = p_c*p_r
            cv = tf.reduce_mean(cv,axis=1)         
            depth = 0
            shaped =(1,(shape[1]//self.block_size),(shape[2]//self.block_size),1)
            temp = tf.zeros(shape=shaped,dtype=tf.float32)
            cv_1 = []
            cv_f = []
            for batch in range(cv.shape[0]): 
                for w in range(cv.shape[1]):
                    for h in range(cv.shape[2]):
                        temp = tf.reshape(cv[batch,w,h,:],shaped)
                        temp = tf.repeat(tf.repeat(temp, self.block_size, axis=1), self.block_size, axis=2)
                        temp = tf.squeeze(temp)
                        cv_1.append(temp)
                cv_f.append(cv_1)
                cv_1 = []
            cv_f = tf.convert_to_tensor(cv_f, dtype=tf.float32)

            cv= tf.transpose(cv_f,[0,2,3,1])
        cv = tf.nn.leaky_relu(cv, 0.1)
        return cv

            

# Optical flow estimator module simple/original -----------------------------------------

# OpticalDlowEstimator -------------------------------------
def _conv_block(filters, kernel_size = (3, 3), strides = (1, 1), batch_norm = False):
  def f(x):
    x = tf.keras.layers.Conv2D(filters, kernel_size,
                         strides, 'same')(x)
    if batch_norm:
      x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, 0.2)
    return x
  return f

class OpticalFlowEstimator(object):
    def __init__(self, name = 'of_estimator'):
        self.batch_norm = False
        #self.name = name

    def __call__(self, cost, x):

        x = tf.concat([cost, x ], axis = 3)
        x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
        x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
        x = _conv_block(96, (3, 3), (1, 1), self.batch_norm)(x)
        x = _conv_block(64, (3, 3), (1, 1), self.batch_norm)(x)
        feature = _conv_block(32, (3, 3), (1, 1), self.batch_norm)(x)
        flow = tf.keras.layers.Conv2D(2, (3, 3), (1, 1), padding = 'same')(feature)

        return flow 


