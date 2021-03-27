from torch import nn
import math
from torchvision import transforms
from build_utils import torch_utils
from build_utils.layers import *
from build_utils.parse_config import *
from PIL import Image
import argparse
import cv2
from torch.autograd import Function
import grad_CAM
from skimage import io
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

ONNX_EXPORT = False


def get_last_conv_name(net):
    """
    :param net:
    :return:
    """
    layer_name = None
    for i, name in enumerate(net.module_list):
        if isinstance(name, nn.Sequential):
            #print(type(name))
            #for idx, m in enumerate(name._modules.items()):
            #    if m[0] == 'Conv2d':
            #        layer_name = m[1]
            #        print(type(m[1]))
            layer_name = name
    return layer_name

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image = image - np.max(np.min(image), 0)
    image = image / np.max(image)
    image = image * 255.
    return np.uint8(image)


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for i, module in enumerate(self.net.module_list):
            if isinstance(name, nn.Sequential):
                for j in range(len(name)):
                    if isinstance(name[j], nn.LeakyReLU):
                        name[j].register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """
        :param module:
        :param grad_in: tuple,
        :param grad_out: tuple,
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],
    :param mask: [H,W],
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb


    #cam = heatmap + np.float32(image)
    image = image / 255
    combined = heatmap * 0.4 + np.float32(image)
    
    combined = (combined * 255).astype(np.uint8)
    
    #cv2.imwrite('./object_map.jpg', combined)
    
    return norm_image(combined), combined


def gen_gb(grad):
    """
    生guided back propagation 
    :param grad: tensor,[3,H,W]
    :return:
    """
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network='yolo', output_dir='./results/'):
    #prefix = os.path.splitext(input_image_name)[0]
    prefix = input_image_name
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), img_as_ubyte(image))


def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: get the structure of each layer based on the .cfg file
    :param img_size:
    :return:
    """

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # remove the first element of cfg list(corresponding to the [net])
    modules_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    # count which layer will be used in the following layers(might be feature fusion or concat)
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # construct each layer
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # if this convolution layer doesn't have bn layer，
                # means that this layer is yolo's predictor
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=False))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            pass

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef["type"] == "yolo":
            yolo_index += 1  # record the number of yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # the scale ratio between prediction feature layer and its raw image

            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = -1
                bias_ = module_list[j][0].bias  # shape(255,) index 0 matches Sequential's Conv2d
                bias = bias_.view(modules.na, -1)  # shape(3, 85)
                bias[:, 4] = bias[:, 4].clone() + (-4.5)  # obj
                bias[:, 5:] = bias[:, 5:].clone() + math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    process output of YOLO layer
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride (corresponding to the raw images) [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # make anchors size scale into grid size
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh,
        # The value corresponding to the dimension with value 1 is not a fixed value, 
        # and subsequent operations can be automatically expanded according to 
        # the broadcast mechanism
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        update grids info and generate new grids parameters
        :param ng: feature map size 
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets construct xy bias of each anchor locased at each cell(on feature map)
        if not self.training:  # boxes training mode doesn't need to regress to final prediction boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy compute the xy coordinates on feature maps
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method compute the wh of feature maps
            io[..., :4] = io[..., :4].clone() * self.stride  # return to raw image size 
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        # this size is only useful if we want to export to ONNX model
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # parse .cfg files
        self.module_defs = parse_model_cfg(cfg)
        # construct each layer based on the modules
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # get all the indices of all the YOLO Layers
        self.yolo_layers = get_yolo_layers(self)

        self.gradients = None
        
        self.features = []

        # print all the information
        self.info(verbose) if not ONNX_EXPORT else None  # print model description
    
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        # yolo_out collects the output of each yolo_layer
        # out collect the outputs
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
                #h = x.register_hook(self.activations_hook)
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
                #h = x.register_hook(self.activations_hook)
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)
                #if i == 88 or i == 100 or i == 112:
                #    x = module(x)
                    #self.features.append(x)
                    #x.register_hook(self.activations_hook)
                    
                    #f_im = tensor_to_PIL(x)
                    
                    #x = x.numpy()
                    #f_im = Image.fromarray(x)
                    #f_im.save("feature_map.jpg")
                #else:
                #    x = module(x)
                    #h = x.register_hook(self.activations_hook)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

            if i == 88:
                x.register_hook(self.activations_hook)
                self.features.append(x)
        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # filter low confidence objs according to objectness
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx cannot support the indices that exceed 1-D
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # filter small object，w > 2 and h > 2 pixel
            # # ONNX cannot suport bitwise_and and all operation
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height filter small obj
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return p
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations_features(self):
        return self.features
    
    #def get_activations(self, x):
    

    def info(self, verbose=False):
        """
        print model information
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)


def get_yolo_layers(self):
    """
    get the indices of three "YOLOLayer" modules
    :param self:
    :return:
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]



