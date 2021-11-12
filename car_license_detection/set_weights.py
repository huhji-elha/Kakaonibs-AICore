import os
import s3_access


def get_files():
    source_bucket = s3_access.get_s3_bucket()
    weights_file = "yolov3_plate_detection_weights.pt"
    os.makedirs("weights/", exist_ok=True)
    weights_save_path = os.path.join(os.getcwd(), "weights", weights_file)
    s3_access.download_file(source_bucket, weights_file, weights_save_path)

    ttf_file = "NanumGothicBold.ttf"
    ttf_save_path = os.path.join(os.getcwd(), "bounding_box", ttf_file)
    s3_access.download_file(source_bucket, ttf_file, ttf_save_path)
