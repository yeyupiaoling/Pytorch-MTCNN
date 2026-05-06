package com.yeyupiaoling.mtcnn;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    // 相机权限请求码
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;
    // 图像分析分辨率设置
    private static final Size ANALYSIS_SIZE = new Size(480, 640);

    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;
    private ExecutorService cameraExecutor;
    private MTCNNDetector mtcnnDetector;
    private FaceOverlayView faceOverlayView;
    
    // 当前摄像头方向：前置或后置
    private int currentLensFacing = CameraSelector.LENS_FACING_FRONT;
    // 防止重复处理同一帧的标志
    private boolean isProcessingFrame = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        
        // 初始化视图组件
        faceOverlayView = findViewById(R.id.faceOverlayView);
        Button switchCameraButton = findViewById(R.id.switchCameraButton);
        
        // 设置切换摄像头按钮的点击事件
        switchCameraButton.setOnClickListener(v -> switchCamera());
        
        // 创建单线程执行器用于相机图像分析
        cameraExecutor = Executors.newSingleThreadExecutor();
        
        // 初始化MTCNN人脸检测器
        mtcnnDetector = new MTCNNDetector(this);
        
        // 检查相机权限
        if (checkCameraPermission()) {
            startCamera();
        } else {
            requestCameraPermission();
        }
    }
    
    // 检查相机权限是否已授予
    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                == PackageManager.PERMISSION_GRANTED;
    }
    
    // 请求相机权限
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this, 
                new String[]{Manifest.permission.CAMERA}, 
                CAMERA_PERMISSION_REQUEST_CODE);
    }
    
    // 处理权限请求结果
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, 
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // 权限请求成功，启动相机
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                // 权限被拒绝，显示提示并退出
                Toast.makeText(this, "相机权限被拒绝", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }
    
    // 启动相机
    private void startCamera() {
        // 获取相机提供者
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
                ProcessCameraProvider.getInstance(this);
        
        // 添加监听器，当相机提供者准备好时绑定用例
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    // 绑定相机用例（预览和图像分析）
    private void bindCameraUseCases() {
        if (cameraProvider == null) {
            return;
        }
        
        // 根据当前选择的摄像头创建CameraSelector
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(currentLensFacing)
                .build();
        
        // 创建预览用例
        Preview preview = new Preview.Builder().build();
        
        // 创建图像分析用例
        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(ANALYSIS_SIZE)
                // 设置背压策略为只保留最新帧，避免处理积压的图像
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        
        // 设置图像分析器
        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeFrame);
        
        // 先解绑所有已绑定的用例
        cameraProvider.unbindAll();
        
        try {
            // 绑定到生命周期
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
            
            // 将预览画面连接到PreviewView
            androidx.camera.view.PreviewView previewView = findViewById(R.id.previewView);
            preview.setSurfaceProvider(previewView.getSurfaceProvider());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    // 分析每一帧图像
    private void analyzeFrame(ImageProxy imageProxy) {
        // 如果正在处理上一帧，跳过这一帧
        if (isProcessingFrame) {
            imageProxy.close();
            return;
        }
        
        isProcessingFrame = true;

        try (imageProxy) {
            // 将ImageProxy转换为Bitmap
            Bitmap bitmap = imageProxyToBitmap(imageProxy);

            if (bitmap != null) {
                // 使用MTCNN检测人脸
                long startTime = System.currentTimeMillis();
                List<FaceData> faces = mtcnnDetector.detect(bitmap);
                Log.d(TAG, "检测到的人脸耗时: " + (System.currentTimeMillis() - startTime) + "ms，结果: " + faces);

                // 在UI线程更新人脸覆盖视图
                int sourceImageWidth = bitmap.getWidth();
                int sourceImageHeight = bitmap.getHeight();
                runOnUiThread(() -> faceOverlayView.setFaces(faces, sourceImageWidth, sourceImageHeight));

                // 回收Bitmap内存
                bitmap.recycle();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // 重置处理标志并关闭ImageProxy
            isProcessingFrame = false;
        }
    }
    
    // 将ImageProxy（YUV格式）转换为Bitmap
    private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        
        if (planes.length < 3) {
            return null;
        }
        
        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();
        
        // 获取 YUV 三个平面的数据
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        byte[] yBytes = new byte[yBuffer.remaining()];
        byte[] uBytes = new byte[uBuffer.remaining()];
        byte[] vBytes = new byte[vBuffer.remaining()];
        yBuffer.get(yBytes);
        uBuffer.get(uBytes);
        vBuffer.get(vBytes);

        int yRowStride = planes[0].getRowStride();
        int yPixelStride = planes[0].getPixelStride();
        int uRowStride = planes[1].getRowStride();
        int uPixelStride = planes[1].getPixelStride();
        int vRowStride = planes[2].getRowStride();
        int vPixelStride = planes[2].getPixelStride();

        // CameraX 的 YUV_420_888 可能包含 stride，这里按 stride 正确还原每个像素
        int[] argb = new int[width * height];
        for (int y = 0; y < height; y++) {
            int yRowOffset = y * yRowStride;
            int uvRowOffset = (y / 2);
            int uRowOffset = uvRowOffset * uRowStride;
            int vRowOffset = uvRowOffset * vRowStride;

            for (int x = 0; x < width; x++) {
                int yValue = yBytes[yRowOffset + x * yPixelStride] & 0xff;
                int uValue = uBytes[uRowOffset + (x / 2) * uPixelStride] & 0xff;
                int vValue = vBytes[vRowOffset + (x / 2) * vPixelStride] & 0xff;

                // 使用标准 YUV 转 RGB 公式，避免颜色偏差影响检测
                int c = Math.max(0, yValue - 16);
                int d = uValue - 128;
                int e = vValue - 128;

                int r = clampToByte((298 * c + 409 * e + 128) >> 8);
                int g = clampToByte((298 * c - 100 * d - 208 * e + 128) >> 8);
                int b = clampToByte((298 * c + 516 * d + 128) >> 8);

                argb[y * width + x] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
        
        // 创建Bitmap
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(argb, 0, width, 0, 0, width, height);
        
        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
        // 分析图像需要和预览方向保持一致，否则竖屏下送入模型的人脸可能是侧着的
        if (rotationDegrees != 0 || currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
            Matrix matrix = new Matrix();
            // 先按传感器返回的角度旋转到正确方向
            if (rotationDegrees != 0) {
                matrix.postRotate(rotationDegrees);
            }
            // 前置摄像头再做水平镜像，保持和预览一致
            if (currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
                float centerX = rotationDegrees % 180 == 0 ? width / 2f : height / 2f;
                float centerY = rotationDegrees % 180 == 0 ? height / 2f : width / 2f;
                matrix.postScale(-1, 1, centerX, centerY);
            }
            Bitmap transformedBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
            bitmap.recycle();
            return transformedBitmap;
        }
        
        return bitmap;
    }

    // 将颜色值限制到 0-255 范围内，避免 YUV 转换后的溢出
    private int clampToByte(int value) {
        return Math.max(0, Math.min(255, value));
    }
    
    // 切换前置和后置摄像头
    private void switchCamera() {
        if (cameraProvider == null) {
            return;
        }
        
        // 切换摄像头方向
        currentLensFacing = (currentLensFacing == CameraSelector.LENS_FACING_FRONT) 
                ? CameraSelector.LENS_FACING_BACK 
                : CameraSelector.LENS_FACING_FRONT;
        
        // 重新绑定相机用例
        bindCameraUseCases();
    }
    
    // 资源释放
    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 释放MTCNN检测器资源
        if (mtcnnDetector != null) {
            mtcnnDetector.release();
        }
        // 关闭执行器
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }
}
