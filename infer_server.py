import os
import io
import time
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from utils.utils import generate_bbox, py_nms, convert_to_square
from utils.utils import pad, calibrate_box, processed_image

app = FastAPI(
    title="MTCNN 人脸检测 API",
    description="基于 PNet、RNet、ONet 的人脸检测服务",
    version="1.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型路径配置
MODEL_PATH = 'infer_models'

# 加载 PNet 模型
pnet = torch.jit.load(os.path.join(MODEL_PATH, 'PNet.pth'))
pnet.to(device)
softmax_p = torch.nn.Softmax(dim=0)
pnet.eval()

# 加载 RNet 模型
rnet = torch.jit.load(os.path.join(MODEL_PATH, 'RNet.pth'))
rnet.to(device)
softmax_r = torch.nn.Softmax(dim=-1)
rnet.eval()

# 加载 ONet 模型
onet = torch.jit.load(os.path.join(MODEL_PATH, 'ONet.pth'))
onet.to(device)
softmax_o = torch.nn.Softmax(dim=-1)
onet.eval()


class FaceBox(BaseModel):
    """人脸框数据结构"""
    x1: int = Field(..., description="左上角 x 坐标")
    y1: int = Field(..., description="左上角 y 坐标")
    x2: int = Field(..., description="右下角 x 坐标")
    y2: int = Field(..., description="右下角 y 坐标")
    score: float = Field(..., description="置信度分数")


class LandmarkPoint(BaseModel):
    """关键点数据结构"""
    x: int = Field(..., description="关键点 x 坐标")
    y: int = Field(..., description="关键点 y 坐标")


class FaceDetectionResult(BaseModel):
    """单个人脸检测结果"""
    bbox: FaceBox
    landmarks: Optional[List[LandmarkPoint]] = None


class DetectionResponse(BaseModel):
    """检测响应数据结构"""
    face_count: int = Field(..., description="检测到的人脸数量")
    faces: List[FaceDetectionResult] = Field(..., description="人脸检测结果列表")
    image_width: int = Field(..., description="输入图像宽度")
    image_height: int = Field(..., description="输入图像高度")


def predict_pnet(infer_data):
    """使用 PNet 模型预测"""
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def predict_rnet(infer_data):
    """使用 RNet 模型预测"""
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def predict_onet(infer_data):
    """使用 ONet 模型预测"""
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()


def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过 PNet 筛选 box 和 landmark"""
    net_size = 12
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    
    while min(current_height, current_width) > net_size:
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue
        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]
    
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    
    boxes_c = np.vstack([
        all_boxes[:, 0] + all_boxes[:, 5] * bbw,
        all_boxes[:, 1] + all_boxes[:, 6] * bbh,
        all_boxes[:, 2] + all_boxes[:, 7] * bbw,
        all_boxes[:, 3] + all_boxes[:, 8] * bbh,
        all_boxes[:, 4]
    ])
    boxes_c = boxes_c.T
    return boxes_c


def detect_rnet(im, dets, thresh):
    """通过 RNet 筛选 box"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((int(num_boxes), 3, 24, 24), dtype=np.float32)
    
    for i in range(int(num_boxes)):
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except:
            continue
    
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None

    keep = py_nms(boxes, 0.4, mode='Union')
    boxes = boxes[keep]
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c


def detect_onet(im, dets, thresh):
    """通过 ONet 筛选 box 并返回 landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
    
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 128
        cropped_ims[i, :, :, :] = img
    
    cls_scores, reg, landmark = predict_onet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None

    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (
        np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1
    ).T
    landmark[:, 1::2] = (
        np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1
    ).T
    boxes_c = calibrate_box(boxes, reg)

    keep = py_nms(boxes_c, 0.6, mode='Minimum')
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark


def detect_faces(im, min_face_size=20, scale_factor=0.79, pnet_thresh=0.9, rnet_thresh=0.6, onet_thresh=0.7):
    """
    综合使用 PNet、RNet、ONet 进行人脸检测
    
    参数:
        im: 输入图像
        min_face_size: 最小人脸尺寸
        scale_factor: 图像金字塔缩放因子
        pnet_thresh: PNet 置信度阈值
        rnet_thresh: RNet 置信度阈值
        onet_thresh: ONet 置信度阈值
    
    返回:
        boxes_c: 人脸框坐标
        landmark: 人脸关键点
    """
    boxes_c = detect_pnet(im, min_face_size, scale_factor, pnet_thresh)
    if boxes_c is None:
        return None, None
    
    boxes_c = detect_rnet(im, boxes_c, rnet_thresh)
    if boxes_c is None:
        return None, None
    
    boxes_c, landmark = detect_onet(im, boxes_c, onet_thresh)
    return boxes_c, landmark


def process_image(file_bytes: bytes) -> np.ndarray:
    """将上传的文件字节转换为 numpy 数组"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像")
    return img


@app.get("/", tags=["健康检查"])
async def root():
    """服务根路径，返回服务状态"""
    return {"status": "running", "service": "MTCNN Face Detection API"}


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "pnet": pnet is not None,
            "rnet": rnet is not None,
            "onet": onet is not None
        }
    }


@app.post("/detect", response_model=DetectionResponse, tags=["人脸检测"])
async def detect_faces_api(file: UploadFile = File(..., description="待检测的图像文件")):
    """上传图像进行人脸检测
    
    参数:
        - file: 图像文件 (支持 JPG, PNG 等常见格式)
    
    返回:
        - face_count: 检测到的人脸数量
        - faces: 人脸检测结果列表
        - image_width: 图像宽度
        - image_height: 图像高度
    """
    try:
        contents = await file.read()
        img = process_image(contents)
        h, w = img.shape[:2]
        
        boxes_c, landmarks = detect_faces(img)
        
        if boxes_c is None:
            return DetectionResponse(
                face_count=0,
                faces=[],
                image_width=w,
                image_height=h
            )
        
        faces = []
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            
            face = FaceDetectionResult(
                bbox=FaceBox(
                    x1=int(bbox[0]),
                    y1=int(bbox[1]),
                    x2=int(bbox[2]),
                    y2=int(bbox[3]),
                    score=float(score)
                )
            )
            
            if landmarks is not None and i < landmarks.shape[0]:
                face.landmarks = [
                    LandmarkPoint(x=int(landmarks[i, j * 2]), y=int(landmarks[i, j * 2 + 1]))
                    for j in range(5)
                ]
            
            faces.append(face)
        return DetectionResponse(
            face_count=len(faces),
            faces=faces,
            image_width=w,
            image_height=h
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/detect_base64", tags=["人脸检测"])
async def detect_faces_base64(image_data: dict):
    """
    使用 Base64 编码的图像进行人脸检测
    
    参数:
        - image_data: 包含 base64 编码图像的字典，格式: {"image": "base64字符串"}
    
    返回:
        - 人脸检测结果
    """
    try:
        import base64
        
        if "image" not in image_data:
            raise HTTPException(status_code=400, detail="缺少 'image' 字段")
        
        img_bytes = base64.b64decode(image_data["image"])
        img = process_image(img_bytes)
        h, w = img.shape[:2]
        
        boxes_c, landmarks = detect_faces(img)
        
        if boxes_c is None:
            return {
                "face_count": 0,
                "faces": [],
                "image_width": w,
                "image_height": h
            }
        
        faces = []
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            
            face = {
                "bbox": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3]),
                    "score": float(score)
                }
            }
            
            if landmarks is not None and i < landmarks.shape[0]:
                face["landmarks"] = [
                    {"x": int(landmarks[i, j * 2]), "y": int(landmarks[i, j * 2 + 1])}
                    for j in range(5)
                ]
            
            faces.append(face)
        
        return {
            "face_count": len(faces),
            "faces": faces,
            "image_width": w,
            "image_height": h
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
