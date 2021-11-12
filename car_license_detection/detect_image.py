import os
import json
import s3_access
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from bounding_box import bounding_box as bb


def detect(source=None, save_img=True):
    imgsz = 512
    half = True
    weights = os.path.join(os.getcwd(), "weights", "yolov3_plate_detection_weights.pt")
    out = "/tmp"

    # Initialize
    device = torch_utils.select_device(device="")
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    cfg_path = os.path.join(os.getcwd(), "cfg", "yolov3-spp-custom.cfg")
    model = Darknet(cfg_path, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)["model"])

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        )  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # download s3 image
    source_bucket = s3_access.get_s3_bucket()
    if source:
        file_path = os.path.join("/tmp", source)
        s3_access.download_file(source_bucket, source, file_path)
    else:
        file_path = "sample.png"

    # Set Dataloader
    dataset = LoadImages(file_path, img_size=imgsz)

    # Get names and colors
    names = load_classes("data/class.names")

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != "cpu" else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=True)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres=0.3, iou_thres=0.6, multi_label=False, agnostic=True,
        )

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        h, w = img.shape[2:]
        save_dict = {"result": {"width": w, "height": h, "objects": []}}

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0 = path, "", im0s

            save_path = str(Path(out) / Path(p).name)
            s += "%gx%g " % img.shape[2:]  # print string
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += "%g %ss, " % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)]
                    object_dict = {"attributes": {}}

                    bb.add(im0, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), label)
                    object_dict["attributes"][label] = float(conf)
                    object_dict["left"] = int(xyxy[0]) / w
                    object_dict["top"] = int(xyxy[1]) / h
                    object_dict["right"] = int(xyxy[2]) / w
                    object_dict["bottom"] = int(xyxy[3]) / h
                    save_dict["result"]["objects"].append(object_dict)

            save_dict["result"]["objects"].sort(key=lambda x: x["left"])
            print("%sDone. (%.3fs)" % (s, t2 - t1))

            # # Save results (image with detections)
            if save_img:
                if dataset.mode == "images":
                    path, filename = os.path.split(save_path)
                    origin_name, ext = os.path.splitext(filename)
                    output_file_name = origin_name + "_output" + ext
                    output_save_path = os.path.join(path, output_file_name)
                    cv2.imwrite(output_save_path, im0)

    print("Done. (%.3fs)" % (time.time() - t0))
    json_file_name = origin_name + "_output.json"
    json_save_path = os.path.join(path, json_file_name)
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, ensure_ascii=False)

    s3_access.upload_file(source_bucket, output_save_path, output_file_name)
    s3_access.upload_file(source_bucket, json_save_path, json_file_name)

    return output_file_name
