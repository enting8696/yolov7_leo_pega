import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from models.experimental import attempt_load
from utils.general import non_max_suppression

# 用於存放使用者劃定的危險區域點座標
danger_zone_points = []

# 偵測使用者點擊滑鼠劃定區域座標
def mouse_callback(event, x, y, flags, param):
    global danger_zone_points
    if event == cv2.EVENT_LBUTTONDOWN:
        danger_zone_points.append((x, y))
        print("新增點座標:", (x, y))

#
def postprocess(frame, detections, conf_threshold, danger_zone):
    """
    detections: numpy array, 每筆資料格式為 [x1, y1, x2, y2, conf, cls]
    """
    for detection in detections:
        conf = detection[4]
        class_id = int(detection[5])
        # 僅對 person 進行偵測（index 0）且對目標有足夠的信心
        if conf > conf_threshold and class_id == 0:
            left, top, right, bottom = detection[:4]
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            bbox_polygon = Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
            danger = False
            # 若使用者已劃定危險區域（至少3點）則進行交集判斷
            if danger_zone is not None and len(danger_zone) >= 3:
                danger_polygon = Polygon(danger_zone)
                if danger_polygon.intersects(bbox_polygon):
                    danger = True
            color = (0, 0, 255) if danger else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = "Danger!" if danger else "Safe"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main():
    global danger_zone_points
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用本地的 yolov7.pt 模型檔案，請確保檔案路徑正確
    model_path = "yolov7.pt"
    # 使用 attempt_load 載入本地模型
    model = attempt_load(model_path, map_location=device)
    model.eval()
    
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)
    
    danger_zone = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 如果使用者已劃定危險區域，則將點串組成多邊形並畫出
        if len(danger_zone_points) >= 6:
            danger_zone = danger_zone_points
            cv2.polylines(frame, [np.array(danger_zone, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
        else:
            cv2.putText(frame, "please draw dangerous zone first!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # BGR 轉 RGB，再轉換為 tensor 並正規化
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()
        img /= 255.0  # 正規化至 [0, 1]

        # 調整 tensor 維度 [B, C, H, W]
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 模型推論
        # model(img) 回傳的是一個 tuple，第一個元素是 raw predictions
        with torch.no_grad():
            pred = model(img)[0]
            # 對 raw predictions 使用非極大值抑制
            pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.4)[0]
        
        # 若有檢測結果則轉成 numpy array，否則使用空 list
        detections = pred.cpu().numpy() if pred is not None and len(pred) else []
        
        # 後處理，將繪製 bounding box 並判斷是否落在危險區域內
        frame = postprocess(frame, detections, conf_threshold=0.5, danger_zone=danger_zone)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
        # 按 "r" 鍵可重置危險區域
        if key == ord("r"):
            print("重置危險區域")
            danger_zone_points = []
            danger_zone = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
