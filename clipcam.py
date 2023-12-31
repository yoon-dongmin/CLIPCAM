import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os

from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from utils.dataset import DirDataset
from utils.imagenet_utils import *
from utils.evaluation_tools import *
from utils.preprocess import getAttacker
from utils.grid_utils import *

RESIZE = 1

ImageTransform = getImageTranform(resize=RESIZE) #normalization = True
originalTransform = getImageTranform(resize=RESIZE, normalized=False)

def clipcam(CLIP_MODEL_NAME, CAM_MODEL_NAME, images, sentence, DISTILL_NUM = 0, ATTACK = None, GPU_ID = 'cpu'):
    model, target_layer, reshape_transform = getCLIP( #Resnet계열은 reshape_transform = None
        model_name=CLIP_MODEL_NAME, gpu_id=GPU_ID) 
    # print(target_layer)
    # exit()


    #Grad CAM 호출
    cam = getCAM(model_name=CAM_MODEL_NAME, model=model, target_layer=target_layer,
                gpu_id=GPU_ID, reshape_transform=reshape_transform)
    MASK_THRESHOLD = get_mask_threshold(CLIP_MODEL_NAME) #mask_treshold 설정 높을수록 box크기 작아짐?

    if len(images) == 4:
        final_img = get_clipcam_grid(cam, model, MASK_THRESHOLD, images, sentence, DISTILL_NUM = DISTILL_NUM, ATTACK = ATTACK, GPU_ID = GPU_ID) #distill_num는 기본값 0
    elif len(images) == 1:
        final_img = get_clipcam_single(cam, model, MASK_THRESHOLD, images[0], sentence, DISTILL_NUM = DISTILL_NUM, ATTACK = ATTACK, GPU_ID = GPU_ID)

    del cam
    del model
    return final_img

def get_mask_threshold(CLIP_MODEL_NAME):
    if CLIP_MODEL_NAME == 'RN50':
        MASK_THRESHOLD = 0.5
    if CLIP_MODEL_NAME == 'RN101':
        MASK_THRESHOLD = 0.2
    if CLIP_MODEL_NAME == 'ViT-B/16':
        MASK_THRESHOLD = 0.3
    if CLIP_MODEL_NAME == 'ViT-B/32':
        MASK_THRESHOLD = 0.6
    return MASK_THRESHOLD

def get_clipcam_single(clipcam, model, MASK_THRESHOLD, image, sentence = None, DISTILL_NUM = 0, ATTACK = None, GPU_ID = 'cpu'):
    image = image
    if ATTACK is not None:
        image = image.resize((224, 224))
        image = getAttacker(image, type=ATTACK, gpu_id=GPU_ID)
    orig_image = image
    image = ImageTransform(image) #transformation을 한 이미지
    orig_image = originalTransform(orig_image) #기존 rgb 이미지
    image = image.unsqueeze(0)
    #둘다 gpu로 보냄
    image = image.to(GPU_ID)
    orig_image = orig_image.to(GPU_ID)

    if sentence == None:
        sentence = input(f"Please enter the query sentence: ")
    text  = clip_modified.tokenize(sentence)
    text = text.to(GPU_ID)
    text_features = model.encode_text(text) #텍스트 feature 뽑아옴

    grayscale_cam = clipcam(input_tensor=image, text_tensor=text_features)[0, :]  #[1,224,224] => [224,224]
    # print(grayscale_cam)
    grayscale_cam_total = grayscale_cam[np.newaxis, :] #새로운 차원 추가, 원래대로 #[224,224] => [1,224,224]
    # print(grayscale_cam_total)

    if DISTILL_NUM > 0:
        for distill in range(DISTILL_NUM):
            distill_mask = np.where(grayscale_cam > 0.5, 0, 1)
            distill_mask = torch.tensor(distill_mask).unsqueeze(0)
            distill_mask = torch.cat((distill_mask,distill_mask,distill_mask)).unsqueeze(0)
            distill_mask = distill_mask.to(GPU_ID)
            image = image * distill_mask
            grayscale_cam = clipcam(input_tensor=image, text_tensor=text_features)[0, :]
            grayscale_cam_total += grayscale_cam
        grayscale_cam = (grayscale_cam_total / np.max(grayscale_cam_total))[0, :]
    grayscale_cam_mask = np.where(grayscale_cam_total < MASK_THRESHOLD, 0, 1)
    pred_bbox, pred_mask = MaskToBBox(grayscale_cam_mask, 1)
    final_img = getHeatMapOneBBox(grayscale_cam, orig_image.permute(1, 2, 0).cpu().numpy(), pred_bbox, sentence)
    return final_img

def get_clipcam_grid(clipcam, model, MASK_THRESHOLD, images, sentence = None, DISTILL_NUM = 0, ATTACK = None, GPU_ID = 'cpu'):
    grid = []
    for i in range(4): #이미지 4장에 대해서
        image = images[i]
        image = image.resize((224, 224))
        grid.append(image) #224크기로 만든 다음 붙임
        
    img_grid = get_concat(grid[0], grid[1], grid[2], grid[3])
    image = img_grid.resize((224, 224)) #224로 resize

    if ATTACK is not None: #체크 기본값은 None
        image = image.resize((224, 224))
        image = getAttacker(image, type=ATTACK, gpu_id=GPU_ID)
    orig_image = image

    image = ImageTransform(image)
    orig_image = originalTransform(orig_image)
    image = image.unsqueeze(0)
    image = image.to(GPU_ID)
    orig_image = orig_image.to(GPU_ID)

    if sentence == None:
        sentence = input(f"Please enter the query sentence: ")

    text  = clip_modified.tokenize(sentence)
    text = text.to(GPU_ID)
    text_features = model.encode_text(text)
    grayscale_cam = clipcam(input_tensor=image, text_tensor=text_features)[0, :] #get_cam_weights함수인지 체크
    grayscale_cam_total = grayscale_cam[np.newaxis, :] #체크
    if DISTILL_NUM > 0:
        for distill in range(DISTILL_NUM):
            distill_mask = np.where(grayscale_cam > 0.5, 0, 1)
            distill_mask = torch.tensor(distill_mask).unsqueeze(0)
            distill_mask = torch.cat((distill_mask,distill_mask,distill_mask)).unsqueeze(0)
            distill_mask = distill_mask.to(GPU_ID)
            image = image * distill_mask
            grayscale_cam = clipcam(input_tensor=image, text_tensor=text_features)[0, :]
            grayscale_cam_total += grayscale_cam
        grayscale_cam = (grayscale_cam_total / np.max(grayscale_cam_total))[0, :]
    grayscale_cam = grayscale_cam_total[0, :] #값 확인
    grayscale_cam_mask = np.where(grayscale_cam < MASK_THRESHOLD, 0, 1) #값 확인
    circular_mask = create_circular_mask(h=224, w=224, radius=30)
    grayscale_cam_mask = np.where(circular_mask > 0, 0, grayscale_cam_mask)
    grayscale_cam_img = Image.fromarray(
        (grayscale_cam_mask * 255).astype(np.uint8)).convert('L')

    grayscale_cam_img = grayscale_cam_img.resize((448, 448))
    # grayscale_cam_img.save(os.path.join(SAVE_DIR, str(total_count) + '_graycam.png'))
    grayscale_cam_np = (np.array(grayscale_cam_img) / 255).astype(np.uint8)
    total_pred_mask, total_bboxes = get_4_bbox(grayscale_cam_np)

    final_img = getHeatMapOneBBox(grayscale_cam, orig_image.permute(1, 2, 0).cpu().numpy(), total_bboxes, sentence, size=448)
    return final_img

# img = Image.open('imgs/airplane.jpg')
# final_img = api('ViT-B/16', 'GradCAM', [img], 'an airplane', 0, None)
# final_img.save('test.png')

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="images/1234.jpg",
                    help="single image path or 4 images directory (grid)")
parser.add_argument("--gpu_id", type=str, default='cpu',
                    help="GPU id to work on, \'cpu\'.")
parser.add_argument("--clip_model_name", type=str,
                    default="RN50", help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not")
parser.add_argument("--distill_num", type=int, default=0,
                    help="Number of iterative masking")
parser.add_argument("--attack_type", type=str, default=None, #안개나 눈같은 방해를 해주는 요인
                    help="attack type: \"snow\", \"fog\"")
parser.add_argument("--sentence", type=str, default='Give me the meat',
                    help="input text")
args = parser.parse_args()

if args.gpu_id != 'cpu':
    args.gpu_id = int(args.gpu_id)

#이미지 불러옴
#여러장 어떻게 합치는지 체크
if os.path.isfile(args.image_path):
    img = Image.open(args.image_path)
    images = [img]
else:
    images = []
    for f in os.listdir(args.image_path):
        images.append(Image.open(os.path.join(args.image_path, f)))

final_img = clipcam(args.clip_model_name, args.cam_model_name, images, args.sentence, args.distill_num, args.attack_type, args.gpu_id)
final_img.save('clipcam_output.png')