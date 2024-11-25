import argparse
import time
import cv2
import json


def ssim_opencv(img1, img2):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_index = cv2.quality.QualitySSIM_compute(gray_img1, gray_img2)[0][0]
    return ssim_index


# 加载 Haar Cascade 分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces_in_segment(video, start_frame, end_frame, step, fps, threshold=0.65):
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    detected_faces_count = 0
    total_frames_checked = 0

    for frame_idx in range(start_frame, end_frame, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        total_frames_checked += 1
        if len(faces) > 0:
            detected_faces_count += 1

    # 计算检测到人脸的比例
    detection_ratio = detected_faces_count / total_frames_checked if total_frames_checked > 0 else 0
    return detection_ratio >= threshold  # 返回是否达到了阈值


def detect_faces_in_video(video_path):
    # 开始计时
    start_time = time.time()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # 设置一个默认帧率

    # 初始化变量
    clip_times = []  # 用于存储切片时间的列表
    face_detection_results = []  # 用于存储每个片段中人脸检测的结果
    step = 5  # 定义步长，每隔 5 帧检测一次（片段检测）

    # 进行切片时间点检测
    last_frame = None
    index = 0
    skip_frame_count = int(fps / 10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        index += 1
        if last_frame is None:
            last_frame = frame
            continue

        if index % skip_frame_count == 0:
            r = ssim_opencv(last_frame, frame)
            if r < 0.5:  # 低于阈值时，判定为切片点
                last_frame = frame
                sec = index / fps
                clip_times.append(f'{sec:.2f} s')

    # 检测从0秒到第一个切片点
    if clip_times:
        first_clip_time = float(clip_times[0].split(' ')[0])
        first_segment_end_frame = int(first_clip_time * fps)
        has_significant_face = detect_faces_in_segment(cap, 0, first_segment_end_frame, step, fps)
        face_detection_results.append(f'Face detected in segment 0.00 - {first_clip_time:.2f} s: {has_significant_face}')
        print(f'Face detected in segment 0.00 - {first_clip_time:.2f} s: {has_significant_face}')

    # 检测切片时间点之间的片段
    for i in range(len(clip_times) - 1):
        start_time = float(clip_times[i].split(' ')[0])
        end_time = float(clip_times[i + 1].split(' ')[0])
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        has_significant_face = detect_faces_in_segment(cap, start_frame, end_frame, step, fps)
        segment_info = f'{start_time:.2f} - {end_time:.2f} s'
        face_detection_results.append(f'Face detected in segment {segment_info}: {has_significant_face}')
        print(f'Face detected in segment {segment_info}: {has_significant_face}')

    # 检测最后一个切片点到视频结束的部分
    if clip_times:
        last_clip_time = float(clip_times[-1].split(' ')[0])
        last_clip_frame = int(last_clip_time * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        has_significant_face = detect_faces_in_segment(cap, last_clip_frame, total_frames, step, fps)
        video_duration = total_frames / fps
        face_detection_results.append(f'Face detected in segment {last_clip_time:.2f} - {video_duration:.2f} s: {has_significant_face}')
        print(f'Face detected in segment {last_clip_time:.2f} - {video_duration:.2f} s: {has_significant_face}')

    # 检测整个视频是否有人脸
    frame_interval = int(fps / 5)  # 每秒检测5次（检测频率）

    face_frame_count = 0
    detected_frame_count = 0  # 记录检测的帧数

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开头
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame_number % frame_interval == 0:
            detected_frame_count += 1  # 增加检测帧计数
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                face_frame_count += 1

    # 计算整个视频的面部比例
    if detected_frame_count > 0:
        face_ratio = (face_frame_count / detected_frame_count) * 100
        print(f"Face ratio for entire video = ({face_frame_count} / {detected_frame_count}) * 100 = {face_ratio:.1f}%")
    else:
        face_ratio = 0

    result = "Yes" if face_ratio > 80 else "No"
    print(f"Overall video result: {result}")

    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    # 返回结果
    return result, face_detection_results, clip_times


# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频人脸检测程序")
    parser.add_argument("--input_video", type=str, required=True, help="输入视频的路径")
    parser.add_argument("--output_json", type=str, required=True, help="输出JSON结果的路径")
    args = parser.parse_args()

    result, face_detection_results, clip_times = detect_faces_in_video(args.input_video)

    # 保存为 JSON 文件
    output = {
        'clip_frame_times': clip_times,
        'face_detection_results': face_detection_results,
        'overall_video_result': result
    }

    with open(args.output_json, 'w') as json_file:
        json.dump(output, json_file, indent=4)

    print(f"Results saved to {args.output_json}")
