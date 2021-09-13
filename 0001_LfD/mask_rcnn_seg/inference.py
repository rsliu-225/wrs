import itertools
import os
import pickle
import config
import numpy as np

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from tqdm import tqdm

'''
Configurations
'''

SCORE_THRESHOLD = 0.5
INFERENCE_OUTPUT_DIR = os.path.join(config.ROOT, "mask_rcnn_seg", "inference_results")
if not os.path.exists(INFERENCE_OUTPUT_DIR):
    os.makedirs(INFERENCE_OUTPUT_DIR)

MODEL_SETUP_FILE = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
NUM_CLASSES = 80
DEVICE = 'cpu'


class MaskRcnnPredictor(object):
    def __init__(self):
        register_coco_instances("coco_test2017", {}, "res/image_info_test2017.json", "")

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_SETUP_FILE))

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_SETUP_FILE)
        cfg.MODEL.DEVICE = DEVICE

        self.inference_meta_data = MetadataCatalog.get("coco_test2017")

        self.predictor = DefaultPredictor(cfg)

    def predict(self, image, label=None, device="cpu"):
        predictions = self.predictor(image)["instances"].to(device)

        if label is not None:
            if predictions.has("pred_classes"):
                pred_classes = predictions.get("pred_classes")
                print(pred_classes)
                target_instances_index = pred_classes == label
                target_predictions = Instances(image_size=predictions.image_size)
                for k, v in predictions.get_fields().items():
                    target_predictions.set(k, v[target_instances_index])
                predictions = target_predictions

        return predictions

    def visualize_prediction(self, image, predictions):
        predictions = predictions.to("cpu")
        v = Visualizer(
            image[:, :, ::-1],
            metadata=self.inference_meta_data,
            scale=1.,
            instance_mode=ColorMode.IMAGE_BW
        )
        return v.draw_instance_predictions(predictions).get_image()[:, :, ::-1]


if __name__ == '__main__':
    import local_vis.realsense.realsense as rs

    # data_f_name = config.ROOT + "/img/realsense/seq/A_light.pkl"
    # data = pickle.load(open(data_f_name, 'rb'))
    realsense = rs.RealSense()
    folder_name = "recons"
    depthimglist, rgbimg_list = realsense.load_frame_seq(folder_name)

    label = None
    print("Start inferencing process")

    predictor = MaskRcnnPredictor()
    for i, im in tqdm(enumerate(rgbimg_list)):
        predictions = predictor.predict(im, label)
        print(predictions.get("pred_masks").numpy())
        cv2.imshow("mask", predictions.get("pred_masks").numpy()[0])

        visualized_pred = predictor.visualize_prediction(im, predictions)
        cv2.imshow("prediction", visualized_pred)
        cv2.waitKey(0)
        cv2.imwrite(f'{INFERENCE_OUTPUT_DIR}/{folder_name}/{str(i).zfill(4)}.png', visualized_pred)
