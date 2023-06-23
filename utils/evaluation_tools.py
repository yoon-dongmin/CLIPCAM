import torch
import clip_modified
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def zeroshot_classifier(classnames, templates, model, GPU_ID):
    with torch.no_grad():
        zeroshot_weights = []
        zeroshot_weights_text = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            class_texts = [classname]
            texts = clip_modified.tokenize(texts).to(GPU_ID)  # tokenize
            class_texts = clip_modified.tokenize(
                class_texts).to(GPU_ID)  # tokenize
            class_embeddings = model.encode_text(
                texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            class_text_embeddings = model.encode_text(
                class_texts)  # embed with text encoder
            class_text_embeddings /= class_text_embeddings.norm(
                dim=-1, keepdim=True)
            class_text_embeddings = class_text_embeddings.mean(dim=0)
            class_text_embeddings /= class_text_embeddings.norm()
            zeroshot_weights_text.append(class_text_embeddings)
        class_sentences = torch.stack(zeroshot_weights, dim=0).to(GPU_ID)
        class_words = torch.stack(zeroshot_weights_text, dim=0).to(GPU_ID)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(GPU_ID)
    return zeroshot_weights, class_sentences, class_words


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_indexes = (correct.t() == True).nonzero(as_tuple=False)
    correct_indice_topk = correct_indexes.t()[0]
    correct_indice_top1 = (correct_indexes[:, 1] == 0).nonzero(
        as_tuple=False).t()[0]
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk], [correct_indice_top1, correct_indice_topk]


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))         # Will be zero if both are 0
    SMOOTH = 1e-6
    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou  # Or thresholded.mean() if you are interested in average across the batch


def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou  # Or thresholded.mean()


def getHeatMap(mask, img, filename, bbox, gt_bbox, pred_text=None, gt_text=None):
    img_im = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')
    img_im.save(filename.split('.')[0] + '_ori.png')
    img = np.float32(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    heatmap = np.uint8(255 * cam)
    heatmap_im = Image.fromarray(heatmap).convert('RGB')
    draw = ImageDraw.Draw(heatmap_im, 'RGBA')

    draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]),
                   fill=(0, 0, 0, 0), outline=(255, 0, 0), width=4)
    draw.rectangle((gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]), fill=(
        0, 0, 0, 0), outline=(0, 0, 255), width=4)
    font = ImageFont.truetype("utils/FreeMono.ttf", 18)
    # draw.text((x, y),"Sample Text",(r,g,b))
    if pred_text != None:
        draw.text((bbox[0], bbox[1]), pred_text, (255, 255, 255),
                  font=font, stroke_width=2, stroke_fill=(255, 0, 0))
    if gt_text != None:
        draw.text((gt_bbox[0], gt_bbox[1]), gt_text, (255, 255, 255),
                  font=font, stroke_width=2, stroke_fill=(0, 0, 255))
    heatmap_im.save(filename)


def getHeatMapNoBBox(mask, img, filename):
    img_im = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')
    img_im.save(filename.split('.')[0] + '_ori.png')
    img = np.float32(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    heatmap = np.uint8(255 * cam)
    heatmap_im = Image.fromarray(heatmap).convert('RGB')
    heatmap_im.save(filename)

# def getHeatMapOneBBox(mask, img, bboxes, text=None, size=224):
#     # 이진 mask에서 RGB 이미지로 변환
#     #img_im = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')

#     # img를 float32 형으로 변환
#     img = np.float32(img)

#     # mask를 heatmap으로 변환
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

#     # BGR 형태의 heatmap을 RGB로 변환
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#     # heatmap 정규화
#     heatmap = np.float32(heatmap) / 255

#     # 이미지가 [0,1] 범위에 없으면 예외 발생
#     if np.max(img) > 1:
#         raise Exception("The input image should np.float32 in the range [0, 1]")

#     # heatmap과 원본 이미지 결합
#     cam = heatmap + img

#     # 결합된 이미지를 최대값으로 정규화
#     cam = cam / np.max(cam)

#     # 결합된 이미지를 [0, 255] 범위로 스케일링하고 uint8로 변환
#     heatmap = np.uint8(255 * cam)

#     # RGB 이미지로 변환
#     heatmap_im = Image.fromarray(heatmap).convert('RGB')

#     # 이미지 위에 bounding box 그리기
#     draw = ImageDraw.Draw(heatmap_im, 'RGBA')
#     for bbox in bboxes:
#         draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), fill=(
#             0, 0, 0, 0), outline=(0, 0, 255), width=4)

#     # 이미지 사이즈에 따른 배경 크기 설정
#     if size == 224:
#         bg_size = 350
#     elif size == 448:
#         heatmap_im = heatmap_im.resize((448, 448))
#         bg_size = 600

#     # 배경 이미지 설정
#     img_w, img_h = heatmap_im.size
#     background = Image.new('RGBA', (bg_size, bg_size), (255, 255, 255, 255))
#     bg_w, bg_h = background.size

#     # heatmap 이미지를 배경 이미지의 가운데에 붙이기
#     offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
#     background.paste(heatmap_im, offset)
#     final_img = background

#     # 텍스트를 이미지 위에 그리기
#     font = ImageFont.truetype("utils/FreeMono.ttf", 18)
#     draw = ImageDraw.Draw(final_img, 'RGBA')
#     if size == 224:
#         if text != None:
#             draw.text((50, 294), 'input: ' + text, (255, 255, 255),
#                       font=font, stroke_width=2, stroke_fill=(0, 0, 0))
#             draw.text((140, 40), 'CLIPCAM', (255, 255, 255),
#                       font=font, stroke_width=2, stroke_fill=(0, 0, 0))
#     elif size == 448:
#         if text != None:
#             # 만약 이미지의 크기가 448이고 텍스트가 주어졌다면, 이미지 아래와 위에 텍스트를 추가합니다.
#             draw.text((80, 530), 'input: ' + text, (255, 255, 255),
#                       font=font, stroke_width=2, stroke_fill=(0, 0, 0))
#             # 'CLIPCAM'이라는 문자열을 이미지 상단에 추가합니다.
#             draw.text((260, 50), 'CLIPCAM', (255, 255, 255),
#                       font=font, stroke_width=2, stroke_fill=(0, 0, 0))

#     # 최종적으로 수정된 이미지를 반환합니다.
#     return final_img
def getHeatMapOneBBox(mask, img, bboxes, text=None, size=224):
    # 이미지를 float32 형으로 변환
    img = np.float32(img)

    # 이미지가 [0,1] 범위에 없으면 예외 발생
    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    # 원본 이미지를 [0, 255] 범위로 스케일링하고 uint8로 변환
    img_scaled = np.uint8(255 * img)

    # RGB 이미지로 변환
    img_im = Image.fromarray(img_scaled).convert('RGB')

    # 이미지 위에 bounding box 그리기
    draw = ImageDraw.Draw(img_im, 'RGBA')
    for bbox in bboxes:
        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), fill=(
            0, 0, 0, 0), outline=(0, 0, 255), width=4)

    # 이미지 사이즈에 따른 배경 크기 설정
    if size == 224:
        bg_size = 350
    elif size == 448:
        img_im = img_im.resize((448, 448))
        bg_size = 600

    # 배경 이미지 설정
    img_w, img_h = img_im.size
    background = Image.new('RGBA', (bg_size, bg_size), (255, 255, 255, 255))
    bg_w, bg_h = background.size

    # 원본 이미지를 배경 이미지의 가운데에 붙이기
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img_im, offset)
    final_img = background

    # 텍스트를 이미지 위에 그리기
    font = ImageFont.truetype("utils/FreeMono.ttf", 18)
    draw = ImageDraw.Draw(final_img, 'RGBA')
    if size == 224:
        if text is not None:
            draw.text((50, 294), 'input: ' + text, (255, 255, 255),
                      font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((140, 40), 'CLIPCAM', (255, 255, 255),
                      font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    elif size == 448:
        if text is not None:
            draw.text((80, 530), 'input: ' + text, (255, 255, 255),
                      font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((260, 50), 'CLIPCAM', (255, 255, 255),
                      font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    # 최종적으로 수정된 이미지를 반환합니다.
    return final_img






def MaskToBBox(masks, size):
    # 마스크에서 값이 1인 위치를 찾습니다.
    filtered_cams = np.array(np.where(masks == 1))
    # print(masks.shape)
    # print(filtered_cams.shape)
    # print(len(masks))


    pred_bboxes = []
    # 각 마스크에 대해 반복합니다.
    for index in range(len(masks)):
        # 현재 마스크에서 값이 1인 위치를 찾습니다.
        filtered_cam_cords = filtered_cams.transpose(
        )[np.where(filtered_cams[0] == index)].transpose()
        # print(filtered_cam_cords)
        # print(len(filtered_cam_cords[1]))

        # 해당 마스크에서 object가 있는 위치의 최소 및 최대 x, y 좌표를 찾아 bounding box를 생성합니다.
        # 만약 마스크 내에 object가 없는 경우(즉, 값이 1인 픽셀이 없는 경우), bounding box는 (0, 0, 0, 0)으로 설정됩니다.
        if len(filtered_cam_cords[1]) != 0:
            pred_bbox = np.array((np.min(filtered_cam_cords[2]), np.min(
                filtered_cam_cords[1]), np.max(filtered_cam_cords[2]), np.max(filtered_cam_cords[1])))
        else:
            pred_bbox = np.array((0, 0, 0, 0))

        # 모든 bounding box를 하나의 리스트에 추가합니다.
        pred_bboxes.append(pred_bbox)

    # 리스트를 numpy 배열로 변환합니다.
    pred_bboxes = np.array(pred_bboxes)

    # bounding box에 해당하는 새로운 마스크를 초기화합니다. 이 마스크의 크기는 (size, 224, 224)이며, 모든 값이 0으로 설정됩니다.
    pred_mask = np.zeros((size, 224, 224)).astype('uint8')

    # 각 bounding box에 해당하는 마스크의 위치를 1로 설정합니다.
    # box를 그리기 위함
    for bbox_index in range(len(pred_bboxes)):
        xmin, ymin, xmax, ymax = pred_bboxes[bbox_index][0], pred_bboxes[
            bbox_index][1], pred_bboxes[bbox_index][2], pred_bboxes[bbox_index][3]
        pred_mask[bbox_index][ymin: ymax+1, xmin:xmax+1] = 1

    # bounding box와 그에 해당하는 새로운 마스크를 반환합니다.
    return pred_bboxes, pred_mask
