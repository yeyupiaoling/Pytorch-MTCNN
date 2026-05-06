package com.yeyupiaoling.mtcnn;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;

import java.util.List;

public class FaceOverlayView extends View {
    
    // 绘制人脸框的画笔
    private Paint facePaint;
    // 绘制关键点的画笔
    private Paint landmarkPaint;
    // 绘制置信度文本的画笔
    private Paint textPaint;
    // 人脸数据列表
    private List<FaceData> faces;
    // 模型输入图像的宽度
    private int sourceImageWidth;
    // 模型输入图像的高度
    private int sourceImageHeight;
    
    public FaceOverlayView(Context context) {
        super(context);
        init();
    }
    
    public FaceOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    public FaceOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }
    
    // 初始化画笔
    private void init() {
        // 初始化人脸框画笔
        facePaint = new Paint();
        facePaint.setColor(Color.BLUE);
        facePaint.setStyle(Paint.Style.STROKE);
        facePaint.setStrokeWidth(5f);
        
        // 初始化关键点画笔
        landmarkPaint = new Paint();
        landmarkPaint.setColor(Color.RED);
        landmarkPaint.setStyle(Paint.Style.FILL);
        landmarkPaint.setStrokeWidth(8f);
        
        // 初始化文本画笔
        textPaint = new Paint();
        textPaint.setColor(Color.YELLOW);
        textPaint.setTextSize(40f);
        textPaint.setStyle(Paint.Style.FILL);
    }
    
    // 设置人脸数据
    public void setFaces(List<FaceData> faces) {
        setFaces(faces, 0, 0);
    }

    // 设置人脸数据和模型输入图像尺寸
    public void setFaces(List<FaceData> faces, int sourceImageWidth, int sourceImageHeight) {
        this.faces = faces;
        this.sourceImageWidth = sourceImageWidth;
        this.sourceImageHeight = sourceImageHeight;
        // 触发重绘
        invalidate();
    }
    
    // 绘制人脸框和关键点
    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        
        // 如果没有人脸数据，直接返回
        if (faces == null) {
            return;
        }

        // 如果还没有拿到模型输入图像尺寸，无法做坐标映射，直接返回
        if (sourceImageWidth <= 0 || sourceImageHeight <= 0) {
            return;
        }

        // 如果画布尺寸无效，直接返回
        if (getWidth() <= 0 || getHeight() <= 0) {
            return;
        }

        // PreviewView 默认按铺满居中方式显示，这里使用同样的缩放规则
        float scale = Math.max(getWidth() / (float) sourceImageWidth,
                getHeight() / (float) sourceImageHeight);
        // 计算居中裁剪后的偏移量，保证框和预览画面一致
        float offsetX = (getWidth() - sourceImageWidth * scale) / 2f;
        float offsetY = (getHeight() - sourceImageHeight * scale) / 2f;
        
        // 遍历每个人脸
        for (FaceData face : faces) {
            // 将模型输出的人脸框坐标映射到当前屏幕画布坐标
            float left = face.left * scale + offsetX;
            float top = face.top * scale + offsetY;
            float right = face.right * scale + offsetX;
            float bottom = face.bottom * scale + offsetY;
            
            // 绘制人脸框矩形
            canvas.drawRect(left, top, right, bottom, facePaint);
            
            // 绘制置信度文本
            canvas.drawText(String.format("%.2f", face.score), left, Math.max(40f, top - 10), textPaint);
            
            // 绘制5个关键点
            float[] landmarks = face.landmarks;
            // 只有关键点数量完整时才绘制，避免数组越界
            if (landmarks != null && landmarks.length >= 10) {
                for (int i = 0; i < 5; i++) {
                    // 将模型输出的关键点坐标映射到当前屏幕画布坐标
                    float x = landmarks[i * 2] * scale + offsetX;
                    float y = landmarks[i * 2 + 1] * scale + offsetY;
                    canvas.drawCircle(x, y, 8f, landmarkPaint);
                }
            }
        }
    }
}
