import argparse
import json
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from bounding_box import bounding_box as bb


def detect(save_img=True):
    imgsz = (512, 394) if ONNX_EXPORT else opt.img_size
    print(imgsz)
    out, source, weights, half = (opt.output, opt.source, opt.weights, opt.half)

    # Initialize
    device = torch_utils.select_device(device="cpu" if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

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

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)

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
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            multi_label=False,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
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
                    output_save_path = os.path.join(path, origin_name + "_output" + ext)
                    print("output_save_path : ", output_save_path)
                    cv2.imwrite(output_save_path, im0)

    print("Done. (%.3fs)" % (time.time() - t0))
    with open("conf_result.json", "w", encoding="utf-8") as f:
        json.dump(save_dict, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="cfg/yolov3-spp-custom.cfg", help="*.cfg path")
    parser.add_argument("--names", type=str, default="data/class.names", help="*.names path")
    parser.add_argument(
        "--weights", type=str, default="weights\\best_weigths.pt", help="weights path"
    )
    parser.add_argument(
        "--source", type=str, default="data/samples", help="source"
    )  # input file/folder, 0 for webcam
    parser.add_argument(
        "--output", type=str, default="output", help="output folder"
    )  # output folder
    parser.add_argument("--img-size", type=int, default=512, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IOU threshold for NMS")
    parser.add_argument(
        "--fourcc", type=str, default="mp4v", help="output video codec (verify ffmpeg support)"
    )
    parser.add_argument("--half", action="store_true", help="half precision FP16 inference")
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1) or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file

    with torch.no_grad():
        detect()
