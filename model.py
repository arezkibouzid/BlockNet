import tensorflow as tf
tf.config.run_functions_eagerly(True)
from modulesBlockNet import *


def L2loss(x, y): # shape(# batch, h, w, 2)
    return tf.reduce_mean(tf.reduce_sum(tf.norm(x-y, ord = 2, axis = 3), axis = (1,2)))


def multiscale_loss(flows_gt, flows_pyramid,
                    weights, name = 'multiscale_loss'):
    # Argument flows_gt must be unscaled, scaled inside of this loss function
    # Scale the ground truth flow, stated Sec.4 in the original paper
    flows_gt_scaled = flows_gt/20.
    # Calculate mutiscale loss
    loss = 0.
    for l, (weight, fs) in enumerate(zip(weights, flows_pyramid)):
        # Downsampling the scaled ground truth flow
        _, h, w, _ = tf.unstack(tf.shape(fs))
        fs_gt_down = tf.image.resize(flows_gt_scaled, (h, w),method='nearest')
        # Calculate l2 loss
        loss += weight*L2loss(fs_gt_down, fs)

    return loss




class BLOCKNet(tf.keras.Model):
    def __init__(self, gamma,weights,num_levels, search_range, warp_type,filters
                 ,output_level,name = 'BlockNet'):
        super().__init__()
        self.num_levels = num_levels
        self.s_range = search_range
        self.weightss = weights
        self.gamma = gamma
        self.warp_type = warp_type
        assert output_level < num_levels, 'Should set output_level < num_levels'
        self.output_level = output_level
        #self.name = name
        self.filters = filters
        self.fp_extractor = FeaturePyramidExtractor(self.num_levels, self.filters)
        self.warp_layer = WarpingLayer(self.warp_type)
        self.cv_layer = CostVolumeLayer(self.s_range)
        self.of_estimators = [OpticalFlowEstimator(name = f'optflow_{l}')\
                              for l in range(self.num_levels)]

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        images_0 = x[:,:,:,:3]
        images_1 = x[:,:,:,3:]
        with tf.GradientTape() as tape:
            print("call_")
            finalflow, y_hat, _ = self(images_0,images_1, training=True)  # Forward pass
            # Compute the loss value
            loss = multiscale_loss(y, y_hat, self.weightss)
            l2_losses = [self.gamma * tf.nn.l2_loss(v) for v in self.trainable_variables]
            l2_losses = tf.reduce_sum(l2_losses)
            loss = loss + l2_losses

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, finalflow)
        # Return a dict mapping metric names to current value

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        # Unpack the data
        x, y = data
        images_0 = x[:,:,:,:3]
        images_1 = x[:,:,:,3:]
        # Compute predictions
        finalflow, y_hat, _ = self(images_0,images_1, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_hat, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, finalflow)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
 

    def __call__(self, images_0, images_1,training):

        pyramid_0 = self.fp_extractor(images_0)
        pyramid_1 = self.fp_extractor(images_1)
        flows_forW = []
        flows_pyr = []
        b, h, w, _ = tf.unstack(tf.shape(pyramid_0[0]))
        finalflow = tf.zeros((b, h, w, 2), dtype = tf.float32)
        # coarse to fine processing
        for l, (feature_0, feature_1) in enumerate(zip(pyramid_0, pyramid_1)):
            print(f'Level {l}')
            b, h, w, _ = tf.unstack(tf.shape(feature_0))
                
            if l == 0:
                flow_up = tf.zeros((b, h, w, 2), dtype = tf.float32)*(20/2**(self.num_levels-l))
                #flows.append(flow)
                flows_forW.append(flow_up)
            else:
                flow_up = tf.compat.v1.image.resize_bilinear(flow_base, (h, w))*(20/2**(self.num_levels-l))
                flows_forW.append(flow_up)


            # warping -> costvolume -> optical flow estimation
            feature_0_warped = self.warp_layer(feature_0, flow_up)

            cost = self.cv_layer(feature_1, feature_0_warped)
            flow_base = self.of_estimators[l](cost,feature_1)

            upscale = 2**(self.num_levels-l)
            flows_pyr.append(flow_base)

            # stop processing at the defined level
            if l == self.output_level:
                upscale = 2**(self.num_levels - self.output_level)
                print(f'Finally upscale flow by {upscale}.')
                finalflow = tf.compat.v1.image.resize_bilinear(flow_base, (h*upscale, w*upscale))*20
                return finalflow, flows_pyr, pyramid_0#flows_up, pyramid_0, flows_base, flows_forW
