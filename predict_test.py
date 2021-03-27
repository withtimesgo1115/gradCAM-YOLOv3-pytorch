import os
import json
import torch
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from build_utils import img_utils
from build_utils import torch_utils
from build_utils import utils
from models import Darknet
from draw_box_utils import draw_box
import models
import grad_CAM
from skimage import io
from skimage import img_as_ubyte


def main():
    img_size = 512  #An integer multiple of 32 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # use generated cfg file
    weights = "weights/yolov3spp-29.pt".format(img_size)  # weight file
    json_path = "./data/pascal_voc_classes.json"  # json label file
    img_path = "./img/1.png"
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)
    


    layer_name = models.get_last_conv_name(model)
    #print(layer_name)
    img_o = cv2.imread(img_path)  # BGR
    assert img_o is not None, "Image Not Found " + img_path

    img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
    height, width = img.shape[:2]
    # Convert
    img_rgb = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img_con = np.ascontiguousarray(img_rgb)

    #img_final = torch.as_tensor(img_con[:,:,::-1].astype("float32").transpose(2, 0, 1)).requires_grad_(True)
    img_final = (torch.from_numpy(img_con).to(device).float()).requires_grad_(True)

    img_final = img_final / 255.0  # scale (0, 255) to (0, 1)
    img_final = img_final.unsqueeze(0)  # add batch dimension

    ori_shape = img_o.shape
    final_shape = img_final.shape[2:]


    inputs = {"image": img_final, "height": height, "width": width, "ori_shape": ori_shape, "final_shape": final_shape}

    t1 = torch_utils.time_synchronized()


    grad_cam = grad_CAM.GradCAM(model, layer_name, ori_shape, final_shape)
    mask, box, class_id = grad_cam(inputs)  # cam mask
 
    image_dict = {}
    img = img_o[..., ::-1]
    
    x1, y1, x2, y2 = box
    image_dict['predict_box'] = img[y1:y2, x1:x2]
    image_cam, image_dict['heatmap'] = models.gen_cam(img[y1:y2, x1:x2], mask)

    models.save_image(image_dict, "gradCAM")
 


if __name__ == "__main__":
    main()
