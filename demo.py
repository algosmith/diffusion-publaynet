import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import PIL
import io
from pathlib import Path
from datadings.reader import MsgpackReader
import random
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from diffusioninst.predictor import VisualizationDemo
from diffusioninst import DiffusionInstDatasetMapper, add_diffusioninst_config, DiffusionInstWithTTA
from diffusioninst.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

# constants
WINDOW_NAME = "COCO detections"
root_path="/home/raushan/dataset/"
root_path = Path(root_path)
data_reader = MsgpackReader(root_path / f"publaynet-train.msgpack")

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_diffusioninst_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
    
def get_fiftyone_dicts(index):
    #samples.compute_metadata()

    # dataset_dicts = []
    # for sample in samples.select_fields(["id", "filepath", "metadata", "segmentations"]):
    #     height = sample.metadata["height"]
    #     width = sample.metadata["width"]
    #     record = {}
    #     record["file_name"] = sample.filepath
    #     record["image_id"] = sample.id
    #     record["height"] = height
    #     record["width"] = width

    #     objs = []
    #     for det in sample.segmentations.detections:
    #         tlx, tly, w, h = det.bounding_box
    #         bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
    #         fo_poly = det.to_polyline()
    #         poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
    #         poly = [p for x in poly for p in x]
    #         obj = {
    #             "bbox": bbox,
    #             "bbox_mode": BoxMode.XYWH_ABS,
    #             "segmentation": [poly],
    #             "category_id": 0,
    #         }
    #         objs.append(obj)

    #     record["annotations"] = objs
    #     dataset_dicts.append(record)

    # return dataset_dicts
    print ("index type", type(index))
    sample = data_reader[index]
        # sample["objects"]["bbox_mode"] = BoxMode.XYXY_ABS
    #print ("sample", sample)
    #sample["image"] = cv2.imread(sample["image"]["bytes"])
    sample["image"] = PIL.Image.open(io.BytesIO(sample["image"]["bytes"]))
    image_pil = sample["image"]
    ## convert pil image to numpy array
    sample["image"] = np.array(sample["image"])
    image_np = sample["image"]
    opencv_image = cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR)
    image_shape = (sample['image_height'], sample['image_width'])
    ## convert pil image to numpy array
    print ("image shape", image_shape)
    return opencv_image, image_np, image_shape, sample, image_pil

colors = {'title': (255, 0, 0),
          'text': (0, 255, 0),
          'figure': (0, 0, 255),
          'table': (255, 255, 0),
          'list': (0, 255, 255)}

def draw_ground_truth2(image, annotations, scale_x, scale_y):
    draw = ImageDraw.Draw(image, 'RGBA')
    print("scale x, scale y", scale_x, scale_y)

    # Define colors if not already defined
    colors = {'title': (0, 255, 0)}  # Example: green color for title

    for annotation in annotations:
        # Draw segmentation
        if 'segmentation' in annotation and annotation['segmentation']:
            draw.polygon(annotation['segmentation'][0], fill=colors['title'] + (64,))
        
        # Draw bbox
        x, y, width, height = annotation['bbox']
        x1, y1, x2, y2 = int(x * scale_x), int(y * scale_y), int((x + width) * scale_x), int((y + height) * scale_y)
        #print("x, y, x2, y2", x1, y1, x2, y2)
        draw.rectangle((x1, y1, x2, y2), outline=colors['title'] + (255,), width=2)

        # Draw label
        label_text = 'text'  # Replace 'text' with actual text if available
        w, h = draw.textsize(label_text)
        text_x, text_y = (x1 + 2, y1 + 2) if y2 - y1 > h else (x2 + 2, y1)
        draw.rectangle((text_x, text_y, text_x + w, text_y + h), fill=(64, 64, 64, 255))
        draw.text((text_x, text_y), text=label_text, fill=(255, 255, 255, 255))

    return np.array(image)

def draw_ground_truth(image, annotations, scale_x, scale_y):
    draw = ImageDraw.Draw(image, 'RGBA')
    print ("scale x, scale y", scale_x, scale_y)
    for annotation in annotations:
        # Draw segmentation
        draw.polygon(annotation['segmentation'][0],
                     fill=colors['title'] + (64,))
        # Draw bbox
        x, y, width, height = annotation['bbox']
        x, y, width, height = int(x * scale_x), int(y * scale_y), int(width * scale_x), int(height * scale_y)
        print ("x, y, width, height", x, y, width, height)
        draw.rectangle(
            (x,
             y,
             x + width,
             y + height),
            outline=colors['title'] + (255,),
            width=2
        )
        # Draw label
        w, h = draw.textsize(text='text')
        if height < h:
            draw.rectangle(
                (x + width,
                 y,
                 x + width + w,
                 y + h),
                fill=(64, 64, 64, 255)
            )
            draw.text(
                (x + width,
                 y),
                text='text',
                fill=(255, 255, 255, 255)
            )
        else:
            draw.rectangle(
                (x,
                 y,
                 x + w,
                 y + h),
                fill=(64, 64, 64, 255)
            )
            draw.text(
                (x,
                 y),
                text='text',
                fill=(255, 255, 255, 255)
            )
    return np.array(image)

def scale_bounding_boxes(anns, image_shape_orig) -> None:
    current_size = 256
    scale_x = current_size / image_shape_orig[1]
    scale_y = current_size / image_shape_orig[0] 

    return scale_x, scale_y

def draw_bounding_box(predictions, image_np,image_pil,  image_shape, sample, scale_x, scale_y):
    #image_np = image.cpu().numpy().transpose(1, 2, 0)
    print ("image_np shape: ", image_np.shape)
    # if image_np.dtype == np.float32 or image_np.dtype == np.float64:
    #     img = (image_np * 255).astype(np.uint8)
    img = np.ascontiguousarray(image_np)
    print ("img shape: ", img.shape)
    instances = predictions["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    print ("instances", len(boxes))
    image_pil_copy = image_pil.copy()
    image_numpy = draw_ground_truth2(image_pil, sample["objects"], scale_x, scale_y)
    #plt.imshow(image_numpy)
    #plt.show()
    draw = ImageDraw.Draw(image_pil_copy)

    #exit()
    for box in boxes:
        
        box = box.to("cpu").numpy()
        print ("pred box", box)
        #box = pred.pred_boxes.tensor.cpu().numpy().astype(int)
        
        # img = np.ascontiguousarray(image_np, dtype=np.uint8)
        # background = np.ones_like(img, dtype=np.uint8) * 255  # White background                
        #plt.imshow(image_np)
        #x, y, x2, y2 = int(box[0] * image_np.shape[0] ), int(box[1] * image_np.shape[1]), int(box[2] * image_np.shape[0]), int(box[3] * image_np.shape[1])
        x_min, y_min, x_max, y_max = int(box[0] ) , int(box[1] ), int(box[2]  ), int(box[3]) 
        #x, y, x2, y2 = cx - w//2, cy - h//2, cx + w//2, cy + h//2
        #x, y, x2, y2 = cx, cy, w, h
        # print(cx, cy, w, h)
        #print("cordinates vals", x, y, x2, y2)
        print ("x, y, x2, y2: ", x_min, y_min, x_max, y_max)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)
        

        # print ("x, y, x2, y2: ", x , y, x2, y2)
        # First, draw a slightly larger rectangle in white (or any contrasting color)
        #cv2.rectangle(img, (x-2, y-2), (x2+2, y2+2), (255, 255, 255), thickness=3)
        # Then, draw the original rectangle in black over it
        #cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 0), thickness=2)
        # text_position = (x2 + 10, y)
        # text_position = (int(text_position[0]), int(text_position[1]))
        # cv2.putText(img, 'Text Detected', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    # Convert image back to RGB for matplotlib if needed (OpenCV uses BGR)
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # If you're using Jupyter Notebook, you can display the image with matplotlib
    # import matplotlib.pyplot as plt
    #plt.figure(figsize=(8, 8))  # You can adjust the figure size as needed
    img_pil_np = np.array(image_pil_copy)
    plt.imshow(img_pil_np)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()



        #cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #cv2.putText(img, str(pred.scores.cpu().numpy()), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img
def show_bounding_boxes_original_image(index, img):
    image_path = data_reader[path]["image_file_path"]
    image_path = image_path.split("/")[-1]
    image_path = "/home/raushan/workspace/DiffusionInst/datasets/archive/train-0/publaynet/train/" + image_path
    print ("image path", image_path)
    image, image_np, image_shape, sample, image_pil = get_fiftyone_dicts(path)
    img = cv2.imread(image_path)
    img_pil = Image.open(image_path)
    image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print ("img detail", img.shape)
    #exit()
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    draw_bounding_box(predictions, image_np,img_pil, image_shape, sample)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

    demo = VisualizationDemo(cfg)
    # Generate a random integer between 0 and 10, inclusive
    if args.input:
        if len(args.input) == 1:
            print ("args.input", args.input)
            #index = int(args.input[0])
            #args.input = glob.glob(os.path.expanduser(args.input[0]))
            print ("args.input", args.input)
            #assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            path = 335405
            #exit()
            #image_key = find_key_by_value(data_reader, path)
            image_path = data_reader[path]["image_file_path"]
            image_path = image_path.split("/")[-1]
            image_path = "/home/raushan/workspace/DiffusionInst/datasets/archive/train-0/publaynet/train/" + image_path
            print ("image path", image_path)
            image, image_np, image_shape, sample, image_pil = get_fiftyone_dicts(path)
            img = cv2.imread(image_path)
            img_pil = Image.open(image_path)
            image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = read_image(path, format="BGR")
            
            print ("img detail", img.shape)
            #exit()
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            draw_bounding_box(predictions, image_np,img_pil, image_shape, sample)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    else:
        index = random.randint(0,  4)
        print("Selected index:", index)
        image_dict = data_reader[index]  # Assuming data_reader is a list of image file paths or data
        print ("index", index)
        image, image_np, image_shape, sample, image_pil = get_fiftyone_dicts(index)
        print ("image shape", image_shape)
        scale_x, scale_y = scale_bounding_boxes(sample["objects"], image_shape)
        #print ("scaled_bounding_boxes", scaled_bounding_boxes)
        
        print ("sample", image)
        #exit()
        #img = read_image(path, format="BGR")
        img = image
        print ("img detail", img.shape)
        #exit()
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        draw_bounding_box(predictions, image_np,image_pil, image_shape, sample, scale_x, scale_y)
        logger.info(
            "Random image: detected {} instances in {:.2f}s".format(
                len(predictions["instances"]) if "instances" in predictions else "finished",
                time.time() - start_time
            )
        )
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", visualized_output.get_image()[:, :, ::-1])
        if cv2.waitKey(0) == 27:
            exit()  # esc to quit
        