package com.yeyupiaoling.mtcnn;

import androidx.annotation.NonNull;

import java.util.Arrays;

// 人脸数据结构
public class FaceData {
    // 人脸框坐标
    public float left;
    public float top;
    public float right;
    public float bottom;
    // 置信度分数
    public float score;
    // 5个关键点坐标（x,y交替排列）
    public float[] landmarks;

    public FaceData(float left, float top, float right, float bottom, float score, float[] landmarks) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
        this.score = score;
        this.landmarks = landmarks;
    }

    @NonNull
    @Override
    public String toString() {
        return "FaceData{" + "left=" + left + ", top=" + top + ", right=" + right + ", bottom=" + bottom + ", " +
                "score=" + score + ", landmarks=" + Arrays.toString(landmarks) + '}';
    }
}
