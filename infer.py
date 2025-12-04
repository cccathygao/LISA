import argparse
import json
import os
import sys
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA inference")
    parser.add_argument("--image_id", type=str, help="Single image ID from dataset.json")
    parser.add_argument("--image_list", type=str, help="Path to text file containing image IDs (one per line)")
    parser.add_argument("--dataset_json", default="../referring-segmentation/grefcoco_dataset/dataset.json", type=str, help="Path to dataset JSON file")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./grefcoco_vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def parse_segmentation_string(seg_string):
    """Parse the segmentation string format to extract polygon coordinates."""
    # Extract coordinates from <seg>(x1, y1)(x2, y2)...</seg> format
    pattern = r'<seg>(.*?)</seg>'
    match = re.search(pattern, seg_string)
    if not match:
        return None
    
    coords_string = match.group(1)
    # Extract all coordinate pairs
    coord_pattern = r'\(([^)]+)\)'
    coords = re.findall(coord_pattern, coords_string)
    
    polygon = []
    for coord in coords:
        x, y = coord.split(',')
        polygon.append([float(x.strip()), float(y.strip())])
    
    return np.array(polygon)


def polygon_to_mask(polygon, height, width):
    """Convert polygon coordinates to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon is not None and len(polygon) > 0:
        polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates."""
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return []
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert contour to polygon format
    polygon = []
    for point in largest_contour:
        x, y = point[0]
        polygon.append([float(x), float(y)])
    
    return polygon


def process_single_image(args, model, tokenizer, clip_image_processor, transform, dataset, image_id, mask_output_dir):
    """Process a single image and return results."""
    
    if image_id not in dataset:
        print(f"Error: Image ID '{image_id}' not found in dataset.")
        return None
    
    data_entry = dataset[image_id]
    prompt = data_entry["problem"]
    relative_image_path = data_entry["images"][0]
    image_path = '../referring-segmentation/' + relative_image_path
    gt_answer = data_entry["answer"]
    img_height = data_entry["img_height"]
    img_width = data_entry["img_width"]
    
    print(f"\n{'='*60}")
    print(f"Processing image ID: {image_id}")
    print(f"Prompt: {prompt}")
    print(f"Image path: {image_path}")

    # Process the image
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt_text = prompt_text.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], "")
    prompt_text = conv.get_prompt()

    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    print(f"Model output: {text_output}")

    # Parse ground truth mask
    gt_polygon = parse_segmentation_string(gt_answer)
    gt_mask = polygon_to_mask(gt_polygon, img_height, img_width)

    # Calculate IoU for each predicted mask
    best_iou = 0.0
    best_mask_idx = 0
    best_pred_mask = None
    
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask_np = pred_mask.detach().cpu().numpy()[0]
        pred_mask_binary = pred_mask_np > 0

        # Resize predicted mask to match ground truth dimensions if needed
        if pred_mask_binary.shape != (img_height, img_width):
            pred_mask_binary = cv2.resize(
                pred_mask_binary.astype(np.uint8), 
                (img_width, img_height), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        iou = calculate_iou(pred_mask_binary, gt_mask)
        print(f"IoU for mask {i}: {iou:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            best_mask_idx = i
            best_pred_mask = pred_mask_binary

        # Save visualizations
        # save_path = "{}/{}_mask_{}.jpg".format(
        #     args.vis_save_path, image_id, i
        # )
        # cv2.imwrite(save_path, pred_mask_binary.astype(np.uint8) * 255)

        save_path = "{}/{}_masked_img_{}.jpg".format(
            args.vis_save_path, image_id, i
        )
        save_img = image_np.copy()
        save_img[pred_mask_binary] = (
            image_np * 0.5
            + pred_mask_binary[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask_binary]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)

    print(f"Best IoU: {best_iou:.4f}")
    
    # Save the best predicted mask in JSON format
    if best_pred_mask is not None:
        # Convert mask to polygon
        pred_polygon = mask_to_polygon(best_pred_mask)
        
        # Save best mask
        # mask_img_filename = f"{image_id}_mask.png"
        # mask_img_path = os.path.join(args.vis_save_path, mask_img_filename)
        # cv2.imwrite(mask_img_path, best_pred_mask.astype(np.uint8) * 255)
        
        # Create JSON structure
        output_json = {
            "images": [
                {
                    "id": image_id,
                    "file_path": relative_image_path,
                    "data_source": "https://huggingface.co/datasets/qixiangbupt/grefcoco",
                    "height": img_height,
                    "width": img_width,
                    "scene": "",
                    "is_longtail": False,
                    "task": "referring_segmentation",
                    "problem": prompt,
                    "problem_type": {
                        "num_class": "",
                        "num_instance": ""
                    }
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": image_id,
                    "category_id": None,
                    "bbox": None,
                    "area": None,
                    "shape_type": "polygon",
                    "error_type": None,
                    "iou": float(best_iou),
                    "segmentation": [pred_polygon]
                }
            ]
        }
        
        # Save JSON file
        json_output_path = os.path.join(mask_output_dir, f"{image_id}_mask.json")
        with open(json_output_path, 'w') as f:
            json.dump(output_json, f, indent=4)
        
        print(f"Saved mask JSON to: {json_output_path}")
        return best_iou
    else:
        print("No valid predicted mask found.")
        return None


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)
    mask_output_dir = "./grefcoco_mask_output"
    os.makedirs(mask_output_dir, exist_ok=True)

    # Load dataset JSON
    with open(args.dataset_json, 'r') as f:
        dataset = json.load(f)
    
    # Get list of image IDs to process
    image_ids = []
    if args.image_list:
        # Read from file
        with open(args.image_list, 'r') as f:
            content = f.read()
            # Split by comma and clean up each ID
            image_ids = [id.strip().strip("'\"") for id in content.split(',') if id.strip()]
        print(f"Loaded {len(image_ids)} image IDs from {args.image_list}")
    elif args.image_id:
        # Single image ID
        image_ids = [args.image_id]
    else:
        print("Error: Must provide either --image_id or --image_list")
        return

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    # Process all images
    results = []
    successful = 0
    failed = 0
    
    for idx, image_id in enumerate(image_ids):
        print(f"\n{'#'*60}")
        print(f"Progress: {idx + 1}/{len(image_ids)}")
        print(f"{'#'*60}")
        
        try:
            iou = process_single_image(
                args, model, tokenizer, clip_image_processor, 
                transform, dataset, image_id, mask_output_dir
            )
            if iou is not None:
                results.append((image_id, iou))
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            failed += 1
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {len(image_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if results:
        avg_iou = sum(iou for _, iou in results) / len(results)
        print(f"\nAverage IoU: {avg_iou:.4f}")
        print(f"Best IoU: {max(iou for _, iou in results):.4f}")
        print(f"Worst IoU: {min(iou for _, iou in results):.4f}")
        
        # Save summary
        summary_path = os.path.join(mask_output_dir, "summary.json")
        summary = {
            "total_images": len(image_ids),
            "successful": successful,
            "failed": failed,
            "average_iou": float(avg_iou),
            "results": [{"image_id": img_id, "iou": float(iou)} for img_id, iou in results]
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])