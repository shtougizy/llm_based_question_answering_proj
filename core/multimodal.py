"""
多模态模块：使用 InternVL3.5-2B 从题目图片中提取题目文本
- 若图片含与题目相关的图表/示意图/公式图，则由多模态模型直接完成解题
- 若图片仅含纯文字题目，则只提取题目文本，交由 LLM 解题
- 两种情况只调用模型一次
"""
import logging
import torch
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INTERNVL_MODEL_PATH, INTERNVL_MAX_NEW_TOKENS, DEVICE, TORCH_DTYPE

logger = logging.getLogger(__name__)
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    logger.info(f"加载 InternVL 模型: {INTERNVL_MODEL_PATH}")
    from transformers import AutoTokenizer, AutoModel
    _tokenizer = AutoTokenizer.from_pretrained(
        INTERNVL_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    _model = AutoModel.from_pretrained(
        INTERNVL_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True,
    )
    _model = _model.to(DEVICE)
    _model.eval()
    logger.info("InternVL 模型加载完成")


def _load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    max_size = 1024
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS
        )
    return img


def _preprocess_image(image: Image.Image):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    pixel_values = transform(image).unsqueeze(0)
    if DEVICE == "cuda":
        pixel_values = pixel_values.to(TORCH_DTYPE).cuda()
    return pixel_values


# 一次调用的 prompt：让模型自己判断并选择输出格式
_PROMPT_COMBINED = """请分析这张图片：

第一步：判断图片中除文字以外，是否还包含与题目解答有关的图表、示意图、坐标轴、几何图形、电路图、化学结构式、实验装置图等视觉内容。

如果【有】与题目相关的图表或示意图，请按如下格式输出：
[有图表]
题目：<图片中的题目原文>
解答：<结合图表内容的完整解题过程和答案>

如果【没有】，图片只包含纯文字题目，请按如下格式输出：
[纯文字]
题目：<图片中的题目原文>

注意：
- 坐标轴、几何图、函数图像、实验图、地图等都算"有图表"
- 只有文字、数字、符号的题目算"纯文字"
- 严格按上述格式输出，不要添加其他内容"""


def extract_question_from_image(image_path: str) -> dict:
    """
    从图片中提取题目，并判断是否需要多模态直接解题。

    Returns:
        {
            "question_text": str,       # 题目文本
            "has_figure": bool,         # 是否含图表
            "vl_answer": str or None,   # 多模态直接解答（has_figure=True时有值）
        }
    """
    _load_model()
    image = _load_image(image_path)
    pixel_values = _preprocess_image(image)

    with torch.no_grad():
        response = _model.chat(
            _tokenizer,
            pixel_values=pixel_values,
            question=_PROMPT_COMBINED,
            generation_config=dict(
                max_new_tokens=INTERNVL_MAX_NEW_TOKENS,
                do_sample=False,
            )
        )

    raw = response.strip()
    logger.info(f"InternVL 原始输出: {raw[:150]}...")

    # 解析输出
    has_figure = raw.startswith("[有图表]")
    question_text = ""
    vl_answer = None

    if has_figure:
        # 提取题目和解答
        q_start = raw.find("题目：")
        a_start = raw.find("解答：")
        if q_start != -1 and a_start != -1:
            question_text = raw[q_start + 3:a_start].strip()
            vl_answer = raw[a_start + 3:].strip()
        elif q_start != -1:
            question_text = raw[q_start + 3:].strip()
        else:
            # 格式不符，退化为只提取文字
            has_figure = False
            question_text = raw.replace("[有图表]", "").strip()
    else:
        # 纯文字：提取题目
        q_start = raw.find("题目：")
        if q_start != -1:
            question_text = raw[q_start + 3:].strip()
        else:
            # 模型没按格式输出，把整个输出当题目文本
            question_text = raw.replace("[纯文字]", "").strip()

    if not question_text:
        question_text = raw  # 兜底

    logger.info(f"has_figure={has_figure}, question={question_text[:80]}...")
    if vl_answer:
        logger.info(f"多模态解答: {vl_answer[:80]}...")

    return {
        "question_text": question_text,
        "has_figure": has_figure,
        "vl_answer": vl_answer,
    }
