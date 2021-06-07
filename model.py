import tensorflow as tf
from modulesBlockNet import *


class BLOCKNet(object):
    def __init__(self, num_levels = 3, search_range = 32, warp_type = 'bilinear',
                 output_level = 2, name = 'BlockNet'):
        self.num_levels = num_levels
        self.s_range = search_range
        self.warp_type = warp_type
        assert output_level < num_levels, 'Should set output_level < num_levels'
        self.output_level = output_level
        self.name = name

        self.fp_extractor = FeaturePyramidExtractor_custom(self.num_levels)
        self.warp_layer = WarpingLayer(self.warp_type)
        self.cv_layer = CostVolumeLayer(self.s_range)
        self.of_estimators = [OpticalFlowEstimator_custom(name = f'optflow_{l}')\
                              for l in range(self.num_levels)]
        # self.contexts = ContextNetwork()
        #assert self.context in ['all', 'final'], 'context argument should be all/final'
        #if self.context is 'all':
        #    self.context_nets = [ContextNetwork(name = f'context_{l}')\
        #                         for l in range(self.num_levels)]
        #else:
        #    self.context_net = ContextNetwork(name = 'context')

    def __call__(self, images_0, images_1):
        with tf.compat.v1.variable_scope(self.name) as vs:

            pyramid_0 = self.fp_extractor(images_0)
            pyramid_1 = self.fp_extractor(images_1)
    

            flows_up = []
            flows_base =[]
            flows_forW = []
            flows_final = []
            # coarse to fine processing
            for l, (feature_0, feature_1) in enumerate(zip(pyramid_0, pyramid_1)):
                print(f'Level {l}')
                b, h, w, _ = tf.unstack(tf.shape(feature_0))
                
                if l == 0:
                    flow_up = tf.zeros((b, h, w, 2), dtype = tf.float32)
                    #flows.append(flow)
                    flows_forW.append(flow_up)
                else:
                    #_,h_flow,w_flow,_=tf.unstack(tf.shape(flow))
                    #depth = tf.Variable([h/h_flow,w/w_flow], dtype=tf.float32)
                    flow_up = tf.compat.v1.image.resize_bilinear(flow_base, (h, w))*(1/2**(3-l))
                    flows_forW.append(flow_up)

                #flows_forW.append(flow)

                # warping -> costvolume -> optical flow estimation
                feature_1_warped = self.warp_layer(feature_1, flow_up)

                cost = self.cv_layer(l,feature_0, feature_1_warped)
                flow_base,flow_up = self.of_estimators[l](cost,feature_0)
                flows_base.append(flow_base)

                #, flow)

                # context considering process all/final
                #if self.context is 'all':
                #    flow = self.context_nets[l](feature, flow)
                #elif l == self.output_level: 
                #    flow = self.context_net(feature, flow)

                flows_up.append(flow_up)
                #flows_base.append(flow_1)

                upscale = 2**(self.num_levels-l)
                finalflow_acc = tf.compat.v1.image.resize_bilinear(flow_base, (h*upscale, w*upscale))*upscale
                flows_final.append(finalflow_acc)
                # stop processing at the defined level
                if l == self.output_level:
                    upscale = 2**(self.num_levels - self.output_level)
                    print(f'Finally upscale flow by {upscale}.')
                    finalflow = tf.compat.v1.image.resize_bilinear(flow_base, (h*upscale, w*upscale))*upscale
                    flows_base.append(finalflow)
                    break
                
            return finalflow, flows_final, flows_up, pyramid_0, flows_base, flows_forW

    @property
    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]

