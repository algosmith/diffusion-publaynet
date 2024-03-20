import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
from datadings.reader import MsgpackReader
import numpy as np
import io
import PIL
from pathlib import Path
import sys
sys.path.append("../../")
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from .detector import DiffusionInst
from torchvision.transforms import Compose, Resize


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        
        self.threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST  # workaround

    def run_on_image(self, image, cfg):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        transforms=None,
        image_size=256,
        image2 = image
        ## transform and normalize the image first. 
        diffusion_inst = DiffusionInst(cfg)
        if transforms is not None:
            transforms = transforms
        else:
            transforms = Compose([Resize(image_size)])
        index = 10
        split = "test"
        root_path="/home/raushan/dataset/"
        root_path = Path(root_path)
        data_reader = MsgpackReader(root_path / f"publaynet-{split}.msgpack")
        sample = data_reader[index]
        sample["image"] = PIL.Image.open(io.BytesIO(sample["image"]["bytes"]))
        #sample["image"] = transforms(sample["image"])
        ## convert pil image to numpy array
        sample["image"] = np.array(sample["image"])
        ## change shape
        image = sample["image"]
        image_shape = (sample['image_height'], sample['image_width'])
        #sample["image"] = torch.from_numpy(sample["image"])
        #sample["image"] = torch.from_numpy(sample["image"]).permute(2, 0, 1)
        ## preprocess image begin

        #image_numpy = sample["image"].cpu().numpy()
        images_np = []
        # for x in batched_inputs:
        #     images_np.append(x["image"].cpu().numpy())
        # if image_numpy.shape[0] == 3:
        #     image_numpy = image_numpy.transpose(1, 2, 0)
        #plt.imshow(image_numpy)
        #plt.show()
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, 1, 3).cpu().numpy()
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, 1, 3).cpu().numpy()
        #print ("tensor sizes: ", pixel_mean.size(), pixel_std.size())
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        #self.to(self.device)
        #sample["image"] = sample["image"].unsqueeze(0)
        print ("sample image shape: ", sample["image"].shape)
        
        image = normalizer(sample["image"]) 
        #images = ImageList.from_tensors(images, self.size_divisibility)

        # images_whwh = list()
        # for bi in batched_inputs:
        #     h, w = bi["image"].shape[-2:]
        #     images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        # images_whwh = torch.stack(images_whwh)

        #return images, images_whwh, images_np
        ## end


        #prepared_image = self.predictor.transform_gen.get_transform(image).apply_image(image)
        #image = images [0]
        #prepared_image = diffusion_inst.preprocess_image(image)
        print ("image", image)
        print ("image shape: ", image.shape)
        #image = image[:, :, ::-1]
        predictions = self.predictor(image)
        # Filter
        instances = predictions['instances']
        new_instances = instances[instances.scores > self.threshold]
        predictions = {'instances': new_instances}
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = sample["image"][:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
