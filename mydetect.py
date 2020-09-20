import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from torchvision.ops.boxes import batched_nms

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.general import xywh2xyxy
from utils.torch_utils import select_device, load_classifier, time_synchronized


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def get_grid_index(keep_idx, x):
    dims = [xi[..., 0].numel() for xi in x]
    i = 0
    for dim in dims:
        if keep_idx >= dim:
            keep_idx -= dim
            i += 1
        else:
            break
    return i, unravel_index(keep_idx, x[i][..., 0].shape)[1:]


def get_scale_grid_index(keep_idx, x_shape, x_numel):
    i = 0
    for dim in x_numel:
        if keep_idx >= dim:
            keep_idx -= dim
            i += 1
        else:
            break
    return i, np.unravel_index(keep_idx, x_shape[i])[1:]


def _process_feature(output, img_size, feat_shape,
                     conf_threshold=0.3, iou_threshold=0.3,
                     num_per_scale_features=8,
                     ):
    # input image: [3, 256, 320]
    pred, x, features = output  # pred: [1, 5040,85]
    # x[0]: [1, 3, 32, 40, 85], features[0]: [1, 128, 32, 40]
    # x[1]: [1, 3, 16, 20, 85], features[1]: [1, 256, 16, 20]
    # x[2]: [1, 3, 8, 10, 85], features[2]: [1, 512, 8, 10]
    num_scales = len(features)
    # num_proposals = pred.shape[1]
    num_classes = pred.shape[-1] - 5
    batch_size = pred.shape[0]
    device = pred.device
    feat_shape = [torch.Size([3, img_size//s, img_size//s])
                  for s in [8, 16, 32]]
    shape_numel = torch.tensor([si.numel() for si in feat_shape]).to(device)
    fs_shape = torch.cumsum(shape_numel, 0)
    feat_shape = torch.tensor(feat_shape).to(device)

    """3 steps: 
    - prepare boxes and scores
    - do nms, sort boxes by keep indices
    - take either fixed number of boxes or by some threshold (variable #)
    
    optimization, first batch nms, second, set max 320 candidates,
    third, improve index filtering
    """
    feat_list = [[] for _ in range(num_scales)]
    box_list = [[] for _ in range(num_scales)]
    for i in range(batch_size):
        one_pred = pred[i]
        one_mask = one_pred[..., 4] > conf_threshold  # candidates
        one_mask_idx = torch.nonzero(one_mask, as_tuple=True)[0]
        one_pred_s = one_pred[one_mask]

        # Compute conf
        one_pred_s[:, 5:] *= one_pred_s[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(one_pred_s[:, :4])  # [5040, 4]
        batch_boxes = boxes.unsqueeze(1).expand(
            -1, num_classes, 4)
        one_boxes = batch_boxes.contiguous().view(-1, 4)
        scores = one_pred_s[:, 5:].reshape(-1)
        mask = scores >= conf_threshold
        mask_idx = torch.nonzero(mask, as_tuple=True)[0]
        boxesf = one_boxes[mask]
        scoresf = scores[mask].contiguous()
        # idxsf = idxs[mask].contiguous()
        cols = torch.arange(num_classes, dtype=torch.long)[None, :].to(device)
        num_proposals = one_pred_s.shape[0]
        label_idx = cols.expand(num_proposals, num_classes).reshape(-1)
        labelsf = label_idx[mask]
        keep = batched_nms(boxesf, scoresf, labelsf, iou_threshold)

        proposal_idx, cls_idx = unravel_index(mask_idx, batch_boxes.shape[:-1])
        proposal_idx = proposal_idx[keep]
        proposal_idx = one_mask_idx[proposal_idx]
        cls_idx = cls_idx[keep]
        # cls_idx = one_mask_idx[cls_idx]
        # cls_idx = cls_idx[keep]
        # print('conf filtered num={}, iou_num={}'.format(
        # boxesf.shape, keep.shape))
        boxes /= img_size  # normalize to 0~1
        num_props = len(proposal_idx)
        ss = fs_shape.unsqueeze(1).expand(-1, num_props)
        idx = (proposal_idx//ss).sum(dim=0)
        # x_shape = torch.index_select(feat_shape, 0, idx.long())
        prop_scale_dims = [unravel_index(proposal_idx[idx == nsi],
                                         feat_shape[nsi])[1:]
                           for nsi in range(num_scales)]
        boxes_scale_idx = [keep[idx == nsi] for nsi in range(num_scales)]
        # cls_scale_idx = [cls_idx[idx == nsi] for nsi in range(num_scales)]
        feat = [[features[nsi][i][..., dim1, dim2]
                 for dim1, dim2 in zip(*prop_scale_dims[nsi])]
                for nsi in range(num_scales)]
        box = [boxesf[boxes_scale_idx[nsi]] for nsi in range(num_scales)]
        # cls = [cls_scale_idx[nsi] for nsi in range(num_scales)]
        for ns in range(num_scales):
            if len(feat[ns]) > 0:
                feat_ns = torch.stack(feat[ns][:num_per_scale_features], 0)
                feat_list[ns].append(feat_ns)
                box_list[ns].append(box[ns][:num_per_scale_features])
    for ns in range(num_scales):
        if len(feat_list[ns]) > 0:
            feat_list[ns] = torch.stack(feat_list[ns])
            box_list[ns] = torch.stack(box_list[ns])
            # print('feat_list size:', feat_list[ns].size())
        else:
            feat_list[ns] = None
            box_list[ns] = None
            # print('feat_list size:', feat_list[ns])
    return feat_list, box_list


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    num_classes = len(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # import torchprof
    cuda = device.type != 'cpu'

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)
        feat_shape = [torch.Size([3, 40, 40]),
                      torch.Size([3, 20, 20]),
                      torch.Size([3, 10, 10])]
        feat_list, box_list = _process_feature(pred, imgsz, feat_shape,
                                                opt.conf_thres,
                                                opt.iou_thres)
        # Apply NMS
        pred = non_max_suppression(pred[0], opt.conf_thres,
                                   opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, model, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            num_objects = 0
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    num_objects += n
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%d objects: %s done. (%.3fs)' % (num_objects, s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
    # print(prof.display(show_events=False))
    # trace, event_lists_dict = prof.raw()
    # import pickle
    # with open('profile.pk', 'wb') as f:
    #     pickle.dump(event_lists_dict, f)
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/yolo/yolov5s-vg.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/test_imgs', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='data/test_imgs/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
