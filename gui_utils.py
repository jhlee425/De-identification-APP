import subprocess

segmentation_model = "../weights/yolact_im700_54_800000.pth"
detection_model = "../weights/detection.pt"
face_detection_model = "../weights/face_detection.pt"

def yolact_video_segmentation(video, video_output, score_threshold, topk, video_multiframe, label_dis, classes, deid_level):
    print("YOLACT video segmentation")
    if label_dis == 'display':
        command = r'conda activate deid && cd yolact &&' \
                r'python eval.py ' \
                r'--trained_model={} --score_threshold={} --top_k={} --video_multiframe={} --video={}::{} --classes {} --deid_level {} && exit'\
            .format(segmentation_model, score_threshold, topk, video_multiframe, video, video_output, classes, deid_level)
    elif label_dis == 'undisplay':
        command = r'conda activate yolact && cd yolact &&' \
                r'python eval.py ' \
                r'--trained_model={} --score_threshold={} --top_k={} --video_multiframe={} --video={}::{} --classes {} --deid_level {} --display_bboxes False --display_text False --display_scores False && exit'\
            .format(segmentation_model, score_threshold, topk, video_multiframe, video, video_output, classes, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", video_output)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def yolact_single_image_segmentation(input_image, output_image, score_threshold, topk, label_dis, classes, deid_level):
    print("YOLACT single image segmentation")
    if label_dis == 'display':
        command = r'conda activate deid && cd yolact &&' \
                r'python eval.py ' \
                r'--trained_model={} --score_threshold={} --top_k={} --image={}::{} --classes {} --deid_level {} && exit'\
            .format(segmentation_model, score_threshold, topk, input_image, output_image, classes, deid_level)
    elif label_dis == 'undisplay':
        command = r'conda activate deid && cd yolact &&' \
                r'python eval.py ' \
                r'--trained_model={} --score_threshold={} --top_k={} --image={}::{} --classes {} --deid_level {} --display_bboxes False --display_text False --display_scores False && exit'\
            .format(segmentation_model, score_threshold, topk, input_image, output_image, classes, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", output_image)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def yolact_images_segmentation(images_input_path, images_output_path, score_threshold, topk, label_dis, classes, deid_level):
    print("YOLACT images segmentation")
    if label_dis == 'display':
        command = r'conda activate deid && cd yolact &&' \
                r'python eval.py ' \
                r'--trained_model={} --score_threshold={} --top_k={} --images={}::{} --classes {} --deid_level {} && exit'\
            .format(segmentation_model, score_threshold, topk, images_input_path, images_output_path, classes, deid_level)
    elif label_dis == 'undisplay':
        command = r'conda activate deid && cd yolact &&' \
                r'python eval.py ' \
                r'--trained_model={} --score_threshold={} --top_k={} --images={}::{} --classes {} --deid_level {} --display_bboxes False --display_text False --display_scores False && exit'\
            .format(segmentation_model, score_threshold, topk, images_input_path, images_output_path, classes, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", images_output_path)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def yolar_video_detection(source, output, score_threshold, methods, label, classes, deid_level):
    print("YOLAR video segmentation")
    if methods == 'blur':
        result_path = '../output/video/blurring'
    elif methods == 'mosaic':
        result_path = '../output/video/mosaic'
    elif methods == 'shuffle':
        result_path = '../output/video/shuffle'
    elif methods == 'distortion':
        result_path = '../output/video/distortion'
    command = r'conda activate deid && cd yolor &&' \
              r'python detect.py ' \
              r'--weights {} --source {} --output {} --conf {} --methods {} --label_dis {} --view-img --cfg cfg/yolor_p6.cfg --classes {} --deid_level {} --device 0 && exit'\
        .format(detection_model, source, result_path, score_threshold, methods, label, classes, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", result_path)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def yolar_single_image_detection(source, output, score_threshold, methods, label, classes, deid_level):
    print("YOLAR single image segmentation")
    if methods == 'blur':
        result_path = '../output/image/blurring'
    elif methods == 'mosaic':
        result_path = '../output/image/mosaic'
    elif methods == 'shuffle':
        result_path = '../output/image/shuffle'
    elif methods == 'distortion':
        result_path = '../output/image/distortion'
    command = r'conda activate deid && cd yolor &&' \
              r'python detect.py ' \
              r'--weights {} --source {} --output {} --conf {} --methods {} --label_dis {} --view-img --cfg cfg/yolor_p6.cfg --classes {} --deid_level {} --device 0 && exit'\
        .format(detection_model, source, result_path, score_threshold, methods, label, classes, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", result_path)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def yolar_images_detection(source, output, score_threshold, methods, label, classes, deid_level):
    print("YOLAR images segmentation")
    if methods == 'blur':
        result_path = '../output/image/blurring'
    elif methods == 'mosaic':
        result_path = '../output/image/mosaic'
    elif methods == 'shuffle':
        result_path = '../output/image/shuffle'
    elif methods == 'distortion':
        result_path = '../output/image/distortion'
    command = r'conda activate deid && cd yolor &&' \
              r'python detect.py ' \
              r'--weights {} --source {} --output {} --conf {} --methods {} --label_dis {} --view-img --cfg cfg/yolor_p6.cfg --classes {} --deid_level {} --device 0 && exit'\
        .format(detection_model, source, result_path, score_threshold, methods, label, classes, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", result_path)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def retinaface_video_detection(video_path, save_path, methods, score_threshold, label_dis, deid_level):
    print("RetinaFace video face detection")
    if methods == 'blur':
        save_path = '../output/image/blurring'
    elif methods == 'mosaic':
        save_path = '../output/image/mosaic'
    elif methods == 'shuffle':
        save_path = '../output/image/shuffle'
    elif methods == 'distortion':
        save_path = '../output/image/distortion'
    command = r'conda activate deid && cd retinaface &&' \
              r'python video_detect.py ' \
              r'--video_path {} --save_path {} --model_path {} --methods {} --score_threshold {} --label_dis {} --deid_level {} && exit'\
        .format(video_path, save_path, face_detection_model, methods, score_threshold, label_dis, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", save_path)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def retinaface_single_image_detection(image_path, save_path, methods, score_threshold, label_dis, deid_level):
    print("RetinaFace single image face detection")
    if methods == 'blur':
        save_path = '../output/image/blurring'
    elif methods == 'mosaic':
        save_path = '../output/image/mosaic'
    elif methods == 'shuffle':
        save_path = '../output/image/shuffle'
    elif methods == 'distortion':
        save_path = '../output/image/distortion'
    command = r'conda activate deid && cd retinaface &&' \
              r'python detect.py ' \
              r'--image_path {} --save_path {} --model_path {} --methods {} --score_threshold {} --label_dis {} --deid_level {} && exit'\
        .format(image_path, save_path, face_detection_model, methods, score_threshold, label_dis, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", save_path)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

def retinaface_images_detection(image_path, save_path, methods, score_threshold, label_dis, deid_level):
    print("RetinaFace image face detection")
    if methods == 'blur':
        save_path = '../output/image/blurring'
    elif methods == 'mosaic':
        save_path = '../output/image/mosaic'
    elif methods == 'shuffle':
        save_path = '../output/image/shuffle'
    elif methods == 'distortion':
        save_path = '../output/image/distortion'
    command = r'conda activate deid && cd retinaface &&' \
              r'python detect.py ' \
              r'--image_path {} --save_path {} --model_path {} --methods {} --score_threshold {} --label_dis {} --deid_level {} && exit'\
        .format(image_path, save_path, face_detection_model, methods, score_threshold, label_dis, deid_level)
    print("Command: ", command)
    print("Results will be saved in: ", save_path)

    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True)
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code