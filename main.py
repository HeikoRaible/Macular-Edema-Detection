import os
import ast
import sys
import time
import json
import torch
import shutil
import warnings
import pathlib
import numpy as np
import torchvision.transforms as T

from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings('ignore')


class EdemaDetector:
    def __init__(self, target_path):
        """ initialize EdemaDetector """
        # target path to classify
        self.target_path = target_path
        self.maskrcnn_transform = T.Compose([T.ToTensor()])
        # device
        self.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        # load maskrcnn model
        self.maskrcnn = self._get_segmentation_model()
        self.maskrcnn.to(self.device)
        self.maskrcnn.load_state_dict(torch.load(os.path.join("models", "MRCNN.pth.tar"), map_location=self.device))
        self.maskrcnn.eval()
        self.maskrcnn_threshold = 0.7
        # blacklist
        self.blacklist = set()
        self.blacklist_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "blacklist.txt")
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, "r+") as infile:
                for line in infile.read().splitlines():
                    self.blacklist.add(line)

    def _get_segmentation_model(self, num_classes=2):
        """ builds base segmentation model """
        # load an instance segmentation model pre-trained on COCO
        model = maskrcnn_resnet50_fpn(pretrained=True)
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model
        
    def check(self):
        """ checks for new examinations, classifies and segments their OCT scans """
        # for each examination in target path
        for patient_examination in os.listdir(self.target_path):
            # skip blacklist
            if patient_examination in self.blacklist:
                print(f"{patient_examination} on blacklist. Being skipped..")
                continue
            try:
                # skip non-directories
                if not os.path.isdir(os.path.join(self.target_path, patient_examination)):
                    continue
                patient_examination_split = patient_examination.split("_")
                # sanity check
                if len(patient_examination_split) == 2 and patient_examination_split[0].isnumeric() and patient_examination_split[1].startswith("LOC"):
                    print(f"processing {patient_examination}")
                    examination_path = os.path.join(self.target_path, patient_examination)
                    # prepare result json
                    result_json = {"patient_id": patient_examination_split[0], "examination_id": patient_examination_split[1]}
                    # get images
                    od_images = []
                    os_images = []
                    # for each file in path
                    for filename in os.listdir(examination_path):
                        # if file is an image
                        if filename.startswith("OCT ") and (filename.endswith(".png") or filename.endswith(".jpg")):
                            # load image
                            image = Image.open(os.path.join(examination_path, filename))
                            # sort into respective OD/OS list
                            if filename[len("OCT ")] == "R":
                                od_images.append(image)
                            elif filename[len("OCT ")] == "L":
                                os_images.append(image)
                            else:
                                raise Exception(f"Unexpected input format for {patient_examination} - {filename}!")
                    # segmentation
                    if od_images:
                        od_pxls_cnt = 0
                        for image in od_images:
                            seg_pxls_cnt = self._segment_image(image)
                            od_pxls_cnt += seg_pxls_cnt
                        result_json["OD"] = {"edema": od_pxls_cnt > 0, "size": od_pxls_cnt}
                    if os_images:
                        os_pxls_cnt = 0
                        for image in os_images:
                            seg_pxls_cnt = self._segment_image(image)
                            os_pxls_cnt += seg_pxls_cnt
                        result_json["OS"] = {"edema": os_pxls_cnt > 0, "size": os_pxls_cnt}
                    # save results
                    output_filename = f"{patient_examination}.json"
                    with open(os.path.join(self.target_path, output_filename), "w+") as outfile:
                        json.dump(result_json, outfile, indent=4)
                    # delete dir
                    shutil.rmtree(examination_path)
                    print(f"{patient_examination} processed!")
            except Exception as e:
                print(f"Exception in {patient_examination}:\n{e}\nBeing skipped and added to a blacklist..")
                self.blacklist.add(patient_examination)
                with open(self.blacklist_path, "w+") as outfile:
                    for patient_examination in self.blacklist:
                        outfile.write(f"{patient_examination}\n")
                continue
        print("done checking for new examinations")

    def _segment_image(self, image):
        """ segment image """
        self.maskrcnn_threshold
        # predict
        CLASS_NAMES = ['__background__', 'oedem']
        image = self.maskrcnn_transform(image) 
        image = image.to(self.device)
        pred = self.maskrcnn([image])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > self.maskrcnn_threshold]
        if pred_t: # indexes of masks above self.maskrcnn_threshold
            pred_t = pred_t[-1]
            masks = (pred[0]['masks'] > 0.5).squeeze(dim=1).detach().cpu().numpy()
            pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
            masks = masks[:pred_t + 1]
            pred_boxes = pred_boxes[:pred_t + 1]
            pred_class = pred_class[:pred_t + 1]
        else:
            masks = []
            pred_boxes = []
            pred_class = []
        if len(masks) == 0:
            return 0
        # merge masks and count pixels
        (dim, h, w) = masks.shape
        onemask = np.zeros((dim, h, w))
        for i in range(dim):
            id_nonzero = masks[i, :, :] > 0
            onemask[0, id_nonzero] = 1
            helper = onemask[0, :, :]
            seg_pxls_cnt = np.count_nonzero(helper)
        return seg_pxls_cnt


if __name__ == "__main__":
    # sanity checks
    if len(sys.argv) != 3:
        print("path to target directory and desired checking interval in minutes required!")
        print("try again using this format: python main.py TARGET_PATH CHECKING_INTERVAL")
        sys.exit()
    target_path = sys.argv[1]
    checking_interval = sys.argv[2]
    if not os.path.exists(target_path):
        print(f"target path {target_path} doesn't exist! try again..")
        sys.exit()
    if not checking_interval.isnumeric():
        print(f"checking interval {checking_interval} isn't a number! try again..")
        sys.exit()
    checking_interval = int(checking_interval)
    # edema detector
    edema_detector = EdemaDetector(target_path)
    # check for new examinations every checking_interval minutes
    while True:
        try:
            edema_detector.check()
        except Exception as e:
            print(f"Exception:\n{e}\ntrying again in {checking_interval} minutes..")
        finally:
            print(f"sleeping {checking_interval} minutes..")
            time.sleep(checking_interval*60)
