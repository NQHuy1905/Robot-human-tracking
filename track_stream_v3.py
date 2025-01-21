import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import cv2
from time import time
from time import sleep
import datetime


import sys
import platform
import numpy as np
from pathlib import Path
import torch

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pandas as pd
from scipy.spatial.distance import cdist

app = FaceAnalysis(providers=[('CUDAExecutionProvider')])
app.prepare(ctx_id=0, det_size=(640, 640))

################################################
import logging
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, \
    process_mask_native
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker
from utils.data_transfer import DictionarySender
from utils.milvus_tool import *
from utils.process_box import *
from utils.camera_setup import GstreamerPython, RealSense
from gi.repository import Gst

def find_x_extremes(masks, bboxes, img1_shape, img0_shape):

    gainy = img1_shape[0] / img0_shape[0] 
    gainx = img1_shape[1] / img0_shape[1]
    results = []
    for mask, bbox in zip(masks, bboxes):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate the y-coordinate at the center of the bounding box
        center_y = int((y2+y1)/2)
        
        # Get the row of the mask at center_y
        mask_row = mask[center_y, x1:x2]

        
        if len(mask_row) > 0:
            # Find the leftmost (smallest x) and rightmost (biggest x) non-zero pixels
            non_zero_indices = np.nonzero(mask_row)[:,0]

            smallest_x = x1 + non_zero_indices[0]
            biggest_x = x1 + non_zero_indices[-1]
            # Scale back to original image coordinates
            smallest_x_original = int(smallest_x / gainx)
            biggest_x_original = int(biggest_x / gainx)
            center_x_original = int((smallest_x_original + biggest_x_original) / 2)
            center_y_original = int(center_y / gainy)
            results.append([center_x_original, center_y_original])

        else:
            results.append(None)
    return results

def normalize(emb):
    emb = np.squeeze(emb)
    norm = np.linalg.norm(emb)
    return emb / norm


def pre_transform(im, imgsz, model):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(imgsz, auto=same_shapes and model.pt, stride=model.stride)
    return [letterbox(image=x) for x in im]

def process_faces(img, collection):
    face_frame = img
    faces = app.get(face_frame)
    for face in faces:
        query_vectors = np.array([normalize(face.embedding)])  
        import math
        # Perform a vector search using IP metric
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_vectors,  # Use the first embedding as the query
            anns_field="embedding",
            param=search_params,
            limit=1,
            expr=None,
            output_fields=["name"]
        )

        th = 0.6
        pred_name, score = results[0][0].entity.get('name'), results[0][0].distance
        if score >= th:
            face["ID_face"] = pred_name
        else:
            face["ID_face"] = None
    faces_data = pd.DataFrame([face.bbox for face in faces], columns= ['x1', 'y1', 'x2', 'y2'])
    faces_data['ID_face'] = [face.ID_face for face in faces]
    return faces_data

def process_detection(dt, im0s, paths, model, device, imgsz, save_dir, is_seg, conf_thres, iou_thres, max_det, augment, visualize, classes, agnostic_nms):
    with dt[0]:
        not_tensor = not isinstance(im0s, torch.Tensor)
        if not_tensor:
            im = np.stack(pre_transform(im0s, imgsz, model))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

            im = im.to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if not_tensor:
                im /= 255  # 0 - 255 to 0.0 - 1.0

    # Inference
    with dt[1]:
        visualize = (
                    increment_path(save_dir / Path(paths).stem, mkdir=False)
                    if visualize #and (not source_type.tensor)
                    else False
                    )
        preds = model(im, augment=augment, visualize=visualize, embed=None)  
    # Apply NMS
    with dt[2]:
        if is_seg:
            masks = []
            # p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            p = ops.non_max_suppression(
                                        preds[0],
                                        conf_thres,
                                        iou_thres,
                                        agnostic=agnostic_nms,
                                        max_det=max_det,
                                        nc=len(model.names),
                                        classes=classes,
                                    )
            proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
            return p, im, masks, proto
        else:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            return p, im, None, None

def process_tracking(p, dt, masks, proto, im0s, paths, curr_frames, 
                     prev_frames, tracker_list, outputs, 
                     faces_data, depth_frame, display_center, 
                     f_pixel, known_id, track_lost_buffer,
                     save_crop, line_thickness, names,
                     im, DEPTH_WIDTH, DEPTH_HEIGHT,
                     save_vid, show_vid, hide_labels,
                     hide_conf, hide_class, windows,
                     seen, pre_velocities_cal, DELTA_T,
                     id_to_send, face_detect_flag, sender
                     ):
    rs_dicts = []
    curr_velocities = []
    # Process detections
    for i, det in enumerate(p):  # detections per image       
        seen += 1
        p, im0, _ = paths[i], im0s[i].copy(), 0#dataset.count
        p = Path(p)  # to Path
        curr_frames[i] = im0

        # s += '%gx%g ' % im.shape[2:]  # print string
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
                
        if det is not None and len(det):
            # track_lost_buffer = 0
            mask = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            x_extremes = find_x_extremes(mask, det[:, :4], im.shape[2:], im0.shape)
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
            # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # pass detections to strongsort
            with dt[3]:
                outputs[i] = tracker_list[i].update(det.cpu(), im0, x_extremes)
            # draw boxes for visualization
            if len(outputs[i]) > 0:

                human_data = pd.DataFrame([output for output in outputs[i]], columns=['x1', 'y1', 'x2', 'y2', 'ID', 'cls', 'conf', 'center_point'])
                human_data['ID_face'] = None  # Initialize the new column
                if not faces_data.empty and len(human_data) == len(faces_data):
                    # Calculate IoU
                    faces_boxes = faces_data[['x1', 'y1', 'x2', 'y2']].values
                    human_boxes = human_data[['x1', 'y1', 'x2', 'y2']].values
                    iou_matrix, center_dist_matrix = calculate_iou_and_center_distance(faces_boxes, human_boxes)
                    # # Find the closest bounding box in group B for each box in A
                    # closest_B_indices = np.argmax(iou_matrix, axis=1)
                    # highest_ious = np.max(iou_matrix, axis=1)

                    # Normalize center distances
                    max_dist = np.max(center_dist_matrix)
                    normalized_center_dist = 1 - (center_dist_matrix / max_dist)  # Invert so larger value is better

                    # Combine IoU and normalized center distance
                    weight_iou = 0.6  # Adjust these weights as needed
                    weight_center = 0.4
                    combined_score = weight_iou * iou_matrix + weight_center * normalized_center_dist
                    # Find the closest bounding box in group B for each box in A
                    closest_B_indices = np.argmax(combined_score, axis=1)
                    highest_scores = np.max(combined_score, axis=1)
                    # Add the array from A to B
                    
                    for l, idx in enumerate(closest_B_indices):
                        human_data.at[idx, 'ID_face'] = faces_data.at[l, 'ID_face']

                for j, (output) in human_data.iterrows():#enumerate(outputs[i]):
                    
                #     # ID = None
                    # print(output)
                    bbox = output[['x1', 'y1', 'x2', 'y2']].tolist()#[0:4]
                #     #####################################


                    # print(bbox)
                    human_id = output['ID']#[4]#find_id_matched(output[4],id_dic)  
                    # print(id)
                    cls = output['cls']#[5]
                    conf = output['conf']#[6]
                    if output['ID_face'] is not None :
                            # pass
                            known_id[str(human_id)] = output['ID_face']
                            if output['ID_face'] not in known_id.values():
                                track_lost_buffer +=1
                    id = known_id[str(human_id)] if str(human_id) in known_id else str(human_id)

                    # center = find_Center(bbox)
                    if output['center_point'] is not None:
                        center = output['center_point']
                        center[0] = max(0, min(center[0], DEPTH_WIDTH-1))
                        center[1] = max(0, min(center[1], DEPTH_HEIGHT-1))
                        d = abs(center[0] - display_center)
                        angle = np.arctan(d/f_pixel) #* 180 / np.pi
                        angle = angle if center[0] > display_center else -angle
                        depth = depth_frame.get_distance(int(center[0]), int(center[1]))
                        real_x_position = depth * np.tan(angle)
                    else:
                        print("Fail to segment human")
                        angle = depth = real_x_position = 0
                        center = None
                    
                    # print(output)

                    """
                    ID of box: id from id
                    Box coordinate: (x1, y1, x2, y2) from bbox
                    Depth: depth of bbox from  depth
                    Angle: angle of bbox from center of the screen from a
                    Velocity: velocity of bbox from center of the screen from v        
                    """

                    #if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                    c = int(cls)  # integer class
                    id = str(id)  # integer id
                    label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                            (
                                                                f'{id} {conf:.2f}' if hide_class else f'{id} {conf:.2f} Depth: {depth:.2f}m Angle: {angle:.2f}'))  # (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                    color = colors(c, True)
                    annotator.box_label(bbox, label, center=center, color=color)
                    if not pre_velocities_cal.empty \
                        and not pre_velocities_cal.loc[pre_velocities_cal['ID'] == id].empty:
                        
                        pre_velocity = pre_velocities_cal.loc[pre_velocities_cal['ID'] == id]
                        velocity_x = (pre_velocity['Real_x_position'].values[0] - real_x_position ) / DELTA_T
                        velocity_y = (pre_velocity['Depth'].values[0] - depth) / DELTA_T
                        rs_dict = {"id": id, "bbox": bbox, "depth": depth, "angle": angle, "velocity": [velocity_x, velocity_y]}
                    else:
                        rs_dict = {"id": id, "bbox": bbox, "depth": depth, "angle": angle, "velocity": None}
                    ############################################################    
                    if id == id_to_send:
                        send_dict = {"ID": id, "Depth": depth, "Angle": angle, "BBox": bbox} 
                        # ct stores current time
                        ct = datetime.datetime.now()
                        face_detect_flag = False
                        sender.send_dictionary(send_dict)
                    rs_dicts.append(rs_dict)
                    curr_velocities.append([id, real_x_position, depth])
            
            curr_velocities_cal = pd.DataFrame(curr_velocities, columns=['ID', 'Real_x_position', 'Depth'])
        else:
            curr_velocities_cal = pd.DataFrame(columns=['ID', 'Real_x_position', 'Depth'])
            track_lost_buffer += 1
            
            pass
            # tracker_list[i].tracker.pred_n_update_all_tracks()
        prev_frames[i] = curr_frames[i]
        if track_lost_buffer == 30:
            track_lost_buffer = 0
            known_id = {}
            face_detect_flag = True
            print(face_detect_flag)
        

        # Stream results
        im0 = annotator.result()
        if show_vid:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            # depth_image = np.asanyarray(depth_frame.get_data())
            # depth_image = depth_image.astype(np.uint8)
            # cv2.imshow('Depth', depth_image)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        return rs_dicts, curr_velocities_cal, im0, track_lost_buffer, known_id, face_detect_flag

@torch.no_grad()
def run(
        source='0',
        yolo_weights='yolov5m.pt',  # model.pt path(s),
        reid_weights='osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos"192.168.44.250
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project= 'runs' ,  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        id_to_send = None,
	    multicast = True,
        streaming_host = "0.0.0.0",
	    laser_power_set = 150,
        init_new_data = False

):

    sender = DictionarySender(port = 2011)
    sender.connect()
    id_dic = {}
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Dataloader
    bs = 1
    # dataset = load_inference_source(
    #         source=source,
    #         batch=bs,
    #         vid_stride=vid_stride,
    #         buffer=False,
    #     )
    
    # source_type = dataset.source_type
    # Configure depth and color streams


    COLOR_WIDTH = 1280
    COLOR_HEIGHT = 720
    DEPTH_WIDTH = 1280
    DEPTH_HEIGHT = 720
    FPS = 30
    DELTA_T = 1 / FPS
    f_pixel = (COLOR_WIDTH * 0.5) / np.tan(69 * 0.5 * np.pi / 180)
    display_center = COLOR_WIDTH // 2

    #################################################
    camera_pipeline = RealSense.setup_realsense(DEPTH_WIDTH, DEPTH_HEIGHT, 
                                    COLOR_WIDTH, COLOR_HEIGHT, FPS, laser_power_set)
    if multicast:
        gst_pipeline, appsrc = GstreamerPython.setup_gstreamer_multicast(streaming_host)
    else:
        gst_pipeline, appsrc = GstreamerPython.setup_gstreamer(streaming_host)
    if gst_pipeline is None:
        print("Failed to set up GStreamer pipeline")
        return

    gst_pipeline.set_state(Gst.State.PLAYING)

    #################################################

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(weights=yolo_weights,
                        device=device,
                        dnn=dnn, 
                        fp16=half,
                        batch=bs,
                        fuse=True,
                        verbose=True)
    model.eval()
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs
    
    # Run tracking
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device),
                                Profile(device=device),
                                Profile(device=device),
                                Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    frame_idx = 0 
    collection = init_Milvus(init_new_data)

    known_id = {}
    track_lost_buffer = 0
    face_detect_flag = True
    pre_velocity_cal = pd.DataFrame(columns=['ID', 'Real_x_position', 'Depth'])
    # for frame_idx, batch in enumerate(dataset):
    try:
        while True:
            print("###########################")
            # start = time()
            # frames = camera_pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame()
            # if not depth_frame or not color_frame:
            #     continue
            
            # # Convert images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())

            # im0s = [color_image]
            # paths = source
            # # paths, im0s, s  = batch#path, im, im0s, vid_cap, s = batch

            #  # paths, im0s, s  = batch#path, im, im0s, vid_cap, s = batch
            # if face_detect_flag:
            #     faces_data = process_faces(im0s[0], collection)
            # else:
            #     faces_data = pd.DataFrame()
            # # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False

            # p, im , masks, proto= process_detection(dt, im0s, paths, 
            #                       model, device, imgsz, 
            #                       save_dir, is_seg, conf_thres, 
            #                       iou_thres, max_det, augment, 
            #                       visualize, classes, agnostic_nms
            #                       )
            
            # rs_dicts, pre_velocity_cal, result_frame, track_lost_buffer, known_id, face_detect_flag = process_tracking(p, dt, masks, proto, im0s, paths, curr_frames, 
            #          prev_frames, tracker_list, outputs, 
            #          faces_data, depth_frame, display_center, 
            #          f_pixel, known_id, track_lost_buffer,
            #          save_crop, line_thickness, names,
            #          im, DEPTH_WIDTH, DEPTH_HEIGHT,
            #          save_vid, show_vid, hide_labels,
            #          hide_conf, hide_class, windows,
            #          seen, pre_velocity_cal, DELTA_T
            #          ,id_to_send, face_detect_flag ,sender
            #          )
            # #print(rs_dicts)
            # print("inference time: ", time()-start)

            # if not push_data(appsrc, result_frame):
            #     break

            # # Add a small delay to control the frame rate
            # sleep(1/30)  # Adjust this value to match your desired frame rate

    except KeyboardInterrupt:
        print("Interrupt")
        camera_pipeline.stop()
        gst_pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

    finally:
        # camera_pipeline.stop()
        gst_pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default='ultralyticss-seg.pt',
                        help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default='osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.9, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default= 'runs', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--id-to-send', help='choose the id to send')
    parser.add_argument('--streaming-host', type=str, help='host to stream')
    parser.add_argument('--multicast', default=True, help='Choose mode to stream')
    parser.add_argument('--laser-power-set', type=int, default=True, help='Choose mode to stream')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = f'trackers/{opt.tracking_method}/configs/{opt.tracking_method}.yaml'
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

