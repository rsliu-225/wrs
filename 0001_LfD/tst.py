import numpy as np
import time
import cv2
import detection_utils as du
import utils.vision_utils as vu
import os
import config_LfD as config

def graphcut(image,init_mask,itercont = 10):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliencyMap = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    saliency_mask = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Saliency Mask", saliency_mask)
    cv2.waitKey(0)

    mask = np.where((init_mask == 1) & (saliency_mask == 255), 2, init_mask)
    # mask = np.where((mask == 0) & (saliency_mask == 255), 2, mask)
    # mask[mask == 0] = 2
    cv2.imshow("Mask", mask * 100)
    cv2.waitKey(0)

    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    start = time.time()
    mask, bgModel, fgModel = cv2.grabCut(image, mask, None, bgModel, fgModel, iterCount=itercont,
                                         mode=cv2.GC_INIT_WITH_MASK)
    end = time.time()
    print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))
    values = (
        ("Definite Background", cv2.GC_BGD),
        ("Probable Background", cv2.GC_PR_BGD),
        ("Definite Foreground", cv2.GC_FGD),
        ("Probable Foreground", cv2.GC_PR_FGD),
    )
    # loop over the possible GrabCut mask values
    for (name, value) in values:
        # construct a mask that for the current value
        print("[INFO] showing mask for '{}'".format(name))
        valueMask = (mask == value).astype("uint8") * 255
        # display the mask so we can visualize it
        cv2.imshow(name, valueMask)
        cv2.waitKey(0)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    output = cv2.bitwise_and(image, image, mask=outputMask)

    cv2.imshow("Input", image)
    cv2.imshow("GrabCut Mask", outputMask)
    cv2.imshow("GrabCut Output", output)
    cv2.waitKey(0)

depthimg_list, rgbimg_list, _ = du.load_frame_seq('glue',
                                                  root_path=os.path.join(config.DATA_PATH, 'raw_img/k4a/seq/'))
mask_list, f_name_list = du.load_mask('glue', mask_type='hand')

for i, f_name in enumerate(f_name_list):
    if i < 100:
        continue
    image = rgbimg_list[int(f_name)]
    init_mask = np.asarray(mask_list[i], dtype='uint8')
    graphcut(image, init_mask)

