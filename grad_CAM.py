import cv2
import numpy as np
from build_utils import utils
import torch
from build_utils import img_utils
from matplotlib import pyplot as plt
from PIL import Image
from misc_functions import *


class GradCAM(object):
    """
    1: the network does not update gradient, input requires the update
    2: use targeted class's score to do backward propagation
    """

    def __init__(self, net, layer_name, ori_shape, final_shape):
        self.net = net
        self.layer_name = layer_name
        self.ori_shape = ori_shape
        self.final_shape = final_shape
        self.feature = None
        self.gradient = None
        self.net.eval()
        #self.handlers = []
        #self._register_hook()
        #self.feature_extractor = FeatureExtractor(self.net, self.layer_name)


    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    
    def _register_hook(self):
        for i, module in enumerate(self.net.module_list):
            if module == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                #self.handlers.append(module.register_backward_hook(self._get_grads_hook))
            
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
    
    def imageRev(img):
        #im1 = np.array(img)
        im1 = 255 - im1;
        #im1 = Image.fromarray(im1)
        return im1

    def __call__(self, inputs, index=0):
        
        output = self.net(inputs['image'])[0]
        output_nonmax = utils.non_max_suppression(output, conf_thres=0.25, iou_thres=0.45, multi_label=True)[0]
        print(output_nonmax)
        
        output_nonmax[:, :4] = utils.scale_coords(self.final_shape, output_nonmax[:, :4], self.ori_shape).round()

        
        scores = output_nonmax[:, 4]
        scores = scores.unsqueeze(0)
        print(scores.shape)
        score = torch.max(scores)
        #score = torch.min(scores)
        idx = scores.argmax().numpy()
        one_hot_output = torch.FloatTensor(1, scores.size()[-1]).zero_()
        one_hot_output[0][idx] = 1
        print(one_hot_output)
        print(score)

        self.net.zero_grad()

        scores.backward(gradient=one_hot_output, retain_graph = True)
        

        self.gradient = self.net.get_activations_gradient()

        self.feature = self.net.get_activations_features()

        print(self.gradient)
        #gradient_tensor = torch.tensor(np.array(self.gradient[2]))
        #pooled_gradients = torch.mean(gradient_tensor, dim=[0, 2, 3])
        
        target = self.feature[0].detach().numpy()[0]
        guided_gradients = self.gradient.detach().numpy()[0]
        weights = np.mean(guided_gradients, axis = (1, 2))  # take averages for each gradient
        print(weights.shape)
        print(target.shape)
        # create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype = np.float32)
        
        # multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights-1):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # normalize between 0-1
        # comment this line if use colormap
        cam = 255 - cam
        # comment this line if use pixel matshow, otherwise cancel the comment
        #cam = np.uint8(cam * 255)  # scale between 0-255 to visualize
        
        
        # comment these two lines if use color map
        plt.matshow(cam.squeeze())
        plt.show()
        
        '''
        cam = np.uint8(Image.fromarray(cam).resize((self.ori_shape[1],self.ori_shape[0]), Image.ANTIALIAS))/255
        
        original_image = Image.open('./img/4.png')
        I_array = np.array(original_image)
        original_image = Image.fromarray(I_array.astype("uint8"))
        save_class_activation_images(original_image, cam, 'cam-featuremap')
        '''
        
        ################################## 
        # This is for pixel matplot method
        ##################################
        test_img = cv2.imread('./img/1.png')
        heatmap = cam.astype(np.float32)
        heatmap = cv2.resize(heatmap, (test_img.shape[1], test_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.6 + test_img
        cv2.imwrite('./new_map.jpg', superimposed_img)


        ################################################ 
        # Using these codes here, you can generate CAM map for each object 
        ################################################

        box = output_nonmax[idx][:4].detach().numpy().astype(np.int32)
        #print(box)
        x1, y1, x2, y2 = box
        ratio_x1 = x1 / test_img.shape[1]
        ratio_x2 = x2 / test_img.shape[0]
        ratio_y1 = y1 / test_img.shape[1]
        ratio_y2 = y2 / test_img.shape[0]

        x1_cam = int(cam.shape[1] * ratio_x1)
        x2_cam = int(cam.shape[0] * ratio_x2)
        y1_cam = int(cam.shape[1] * ratio_y1)
        y2_cam = int(cam.shape[0] * ratio_y2)

        cam = cam[y1_cam:y2_cam, x1_cam:x2_cam]
        cam = cv2.resize(cam, (x2 - x1, y2 - y1))
  

        class_id = output[idx][-1].detach().numpy()
        return cam, box, class_id
        














        '''
        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
   
        gradient = self.gradient.cpu().data.numpy()  # [C,H,W]

        weight = np.mean(gradient, axis=(1, 2))  # [C]

        #self.feature = self.net.features
        #print(self.feature)
        #feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]

        #feature = self.feature[idx].cpu().data
        feature = self.feature[0].data.numpy()
        print(pooled_gradients)
        print(torch.any(pooled_gradients != 0))
        print(pooled_gradients.shape)
        print(weight.shape)
        print(feature.shape)


        
        # pool the gradients across the channels
        #pooled_gradients = np.mean(gradient, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.feature[0].detach()
        # weight the channels by corresponding gradients
        for i in range(26):
            #activations[:, i, :, :] *= weight[i]
            activations[:, i, :, :] *= pooled_gradients[i]

        print(activations.shape)
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        plt.matshow(heatmap.squeeze())
        plt.show()

        
        test_img = cv2.imread('./test7.png')
        heatmap = heatmap.numpy().astype(np.float32)
        heatmap = cv2.resize(heatmap, (test_img.shape[1], test_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.6 + test_img
        cv2.imwrite('./map.jpg', superimposed_img)
        '''
        
