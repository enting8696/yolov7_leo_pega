import cv2
import time
import argparse
import random
import numpy as np
import onnxruntime as ort
from pathlib import Path

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def opencv_nms(boxes, scores, score_threshold, nms_threshold):
    """Apply OpenCV NMS to filter overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    # Convert boxes to format expected by cv2.dnn.NMSBoxes (x, y, w, h)
    nms_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        nms_boxes.append([x1, y1, w, h])
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores, score_threshold, nms_threshold)
    
    if len(indices) > 0:
        return indices.flatten()
    else:
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.onnx', help='model.onnx path')
    parser.add_argument('--source', type=str, default='0', help='source (0 for webcam, image file, or video file)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    opt = parser.parse_args()
    print(opt)
    
    # Initialize
    cuda = opt.device != 'cpu'
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(opt.weights, providers=providers)
    
    # Get names for COCO classes
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
            'hair drier', 'toothbrush']
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    
    # Get model input and output names
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    
    # Check source type
    webcam = opt.source.isnumeric() or opt.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Check if source is a video file
    vid_formats = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']
    is_video = any(opt.source.lower().endswith(f'.{fmt}') for fmt in vid_formats)
    
    # Setup input source
    if webcam or is_video:
        cap = cv2.VideoCapture(int(opt.source) if opt.source.isnumeric() else opt.source)
        if not cap.isOpened():
            source_type = "webcam" if webcam else "video"
            print(f"Error: Could not open {source_type} {opt.source}")
            return
        
        if webcam:
            print(f"Successfully opened webcam {opt.source}")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing video: {opt.source} ({total_frames} frames)")
        
        # Set resolution if needed for webcam
        if webcam:
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            pass
    else:
        # Single image
        cap = None
        img = cv2.imread(opt.source)
        if img is None:
            print(f"Error: Could not read image {opt.source}")
            return
    
    t0 = time.time()
    img_size = opt.img_size
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    
    # Frame counter for video
    frame_idx = 0
    
    # Run detection on webcam, video, or image
    if webcam or is_video:
        while True:
            # Read frame from webcam or video
            ret, frame = cap.read()
            if not ret:
                if is_video:
                    print("Finished processing video")
                else:
                    print("Error: Failed to grab frame")
                break
            
            frame_idx += 1
            
            # Process frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            image = img.copy()
            image, ratio, dwdh = letterbox(image, new_shape=img_size, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            im = image.astype(np.float32)
            im /= 255
            
            # Run inference
            t1 = time.time()
            inp = {inname[0]: im}
            outputs = session.run(outname, inp)[0]
            t2 = time.time()
            
            # Apply NMS
            t_nms_start = time.time()
            
            # Filter by confidence
            mask = outputs[:, 6] >= conf_thres
            filtered_outputs = outputs[mask]
            
            # Apply class filter if specified
            if opt.classes is not None:
                class_mask = np.isin(filtered_outputs[:, 5].astype(int), opt.classes)
                filtered_outputs = filtered_outputs[class_mask]
            
            # Apply OpenCV NMS
            if len(filtered_outputs) > 0:
                boxes = filtered_outputs[:, 1:5]  # x1, y1, x2, y2
                scores = filtered_outputs[:, 6]
                classes = filtered_outputs[:, 5]
                
                # Apply NMS
                nms_indices = opencv_nms(boxes, scores, conf_thres, iou_thres)
                
                # Keep only the boxes that survived NMS
                if len(nms_indices) > 0:
                    final_outputs = filtered_outputs[nms_indices]
                else:
                    final_outputs = np.array([])
            else:
                final_outputs = np.array([])
            
            t_nms_end = time.time()
            
            # Process detections
            ori_img = frame.copy()
            
            # Count detections by class
            class_counts = {}
            
            # Draw bounding boxes
            for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(final_outputs):
                # Convert coordinates back to original image
                box = np.array([x0, y0, x1, y1])
                box -= np.array(dwdh * 2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                
                # Get class name and color
                cls_id = int(cls_id)
                score = round(float(score), 3)
                name = names[cls_id]
                color = colors[name]
                
                # Count detections
                class_counts[name] = class_counts.get(name, 0) + 1
                
                # Draw rectangle and text
                label = f'{name} {score}'
                cv2.rectangle(ori_img, box[:2], box[2:], color, 2)
                cv2.putText(ori_img, label, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
            
            # Create detection summary
            if class_counts:
                detection_strings = []
                for cls_name, count in class_counts.items():
                    detection_strings.append(f"{count} {cls_name}{'s' if count > 1 else ''}")
                detection_summary = ', '.join(detection_strings) + ', '
            else:
                detection_summary = ''
            
            # Print time info
            if is_video:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"video 1/1 ({frame_idx}/{total_frames}) {opt.source}: {detection_summary}Done. ({(t2 - t1)*1000:.1f}ms) Inference, ({(t_nms_end - t_nms_start)*1000:.1f}ms) NMS")
            else:
                fps = 1/(t2-t1)
                print(f"webcam: {detection_summary}Done. ({(t2 - t1)*1000:.1f}ms) Inference, ({(t_nms_end - t_nms_start)*1000:.1f}ms) NMS - FPS: {fps:.1f}")
            
            # Display results
            if opt.view_img:
                window_name = "YOLOv7 ONNX Webcam" if webcam else "YOLOv7 ONNX Video"
                cv2.imshow(window_name, ori_img)
                if cv2.waitKey(1) == 27:  # ESC key to quit
                    break
    else:
        # Process single image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        image = img.copy()
        image, ratio, dwdh = letterbox(image, new_shape=img_size, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        
        # Run inference
        t1 = time.time()
        inp = {inname[0]: im}
        outputs = session.run(outname, inp)[0]
        t2 = time.time()
        
        # Apply NMS
        t_nms_start = time.time()
        
        # Filter by confidence
        mask = outputs[:, 6] >= conf_thres
        filtered_outputs = outputs[mask]
        
        # Apply class filter if specified
        if opt.classes is not None:
            class_mask = np.isin(filtered_outputs[:, 5].astype(int), opt.classes)
            filtered_outputs = filtered_outputs[class_mask]
        
        # Apply OpenCV NMS
        if len(filtered_outputs) > 0:
            boxes = filtered_outputs[:, 1:5]  # x1, y1, x2, y2
            scores = filtered_outputs[:, 6]
            classes = filtered_outputs[:, 5]
            
            # Apply NMS
            nms_indices = opencv_nms(boxes, scores, conf_thres, iou_thres)
            
            # Keep only the boxes that survived NMS
            if len(nms_indices) > 0:
                final_outputs = filtered_outputs[nms_indices]
            else:
                final_outputs = np.array([])
        else:
            final_outputs = np.array([])
        
        t_nms_end = time.time()
        
        # Process detections
        ori_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Count detections by class
        class_counts = {}
        
        # Draw bounding boxes
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(final_outputs):
            # Convert coordinates back to original image
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            
            # Get class name and color
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = names[cls_id]
            color = colors[name]
            
            # Count detections
            class_counts[name] = class_counts.get(name, 0) + 1
            
            # Draw rectangle and text
            label = f'{name} {score}'
            cv2.rectangle(ori_img, box[:2], box[2:], color, 2)
            cv2.putText(ori_img, label, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        
        # Create detection summary
        if class_counts:
            detection_strings = []
            for cls_name, count in class_counts.items():
                detection_strings.append(f"{count} {cls_name}{'s' if count > 1 else ''}")
            detection_summary = ', '.join(detection_strings) + ', '
        else:
            detection_summary = ''
        
        # Print time info
        print(f"image 1/1 {opt.source}: {detection_summary}Done. ({(t2 - t1)*1000:.1f}ms) Inference, ({(t_nms_end - t_nms_start)*1000:.1f}ms) NMS")
        
        # Display results
        if opt.view_img:
            cv2.imshow("YOLOv7 ONNX", ori_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Clean up
    if (webcam or is_video) and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    main()