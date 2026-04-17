from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, CLIPModel
from ultralytics import YOLO

from config import get_paths
from utils import image_path_from_id, load_json, save_json

if sys.version_info < (3, 13):
    import faiss  # type: ignore
else:
    faiss = None


def _to_feature_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported CLIP output type: {type(output)}")


def _detect_crop(detector: YOLO, image_path: Path, conf: float = 0.25) -> Image.Image | None:
    results = detector.predict(source=str(image_path), conf=conf, verbose=False)
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    b = boxes[0].xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, b)
    img = Image.open(image_path).convert("RGB")
    x1 = max(0, min(img.width - 1, x1))
    y1 = max(0, min(img.height - 1, y1))
    x2 = max(x1 + 1, min(img.width, x2))
    y2 = max(y1 + 1, min(img.height, y2))
    return img.crop((x1, y1, x2, y2))


def _embed_query(
    model: CLIPModel,
    processor: AutoProcessor,
    device: torch.device,
    crop: Image.Image,
    question: str,
) -> np.ndarray:
    inputs = processor(images=[crop], text=[question], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_vec = _to_feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
        text_vec = _to_feature_tensor(model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ))
        query = F.normalize(image_vec + text_vec, dim=-1)
    return query.cpu().numpy().astype("float32")


def _build_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join(
        [
            f"[Law: {c['law_id']} - Article: {c['article_id']}]\n{c['text']}"
            for c in contexts
        ]
    )
    return (
        "Bạn là trợ lý pháp lý giao thông đường bộ. "
        "Hãy trả lời câu hỏi dựa trên ngữ cảnh luật được truy xuất.\n\n"
        f"Ngữ cảnh:\n{ctx}\n\n"
        f"Câu hỏi: {question}\n"
        "Nếu là trắc nghiệm thì chỉ trả lời 1 ký tự A/B/C/D. "
        "Nếu là đúng sai thì chỉ trả lời Đúng hoặc Sai."
    )


def run_rag(
    root: Path,
    split: str,
    detector_path: Path,
    embedding_model_dir: Path,
    vector_db_dir: Path,
    llm_name: str,
    out_json: Path,
    topk: int = 5,
) -> None:
    paths = get_paths(root)
    if split == "public":
        data = load_json(paths.public_json)
        image_root = paths.public_images
    elif split == "private":
        data = load_json(paths.private_task2_json)
        image_root = paths.private_images
    else:
        raise ValueError("split must be one of: public, private")

    meta = load_json(vector_db_dir / "law_articles_meta.json")
    embeddings = np.load(vector_db_dir / "law_articles.npy").astype("float32")
    use_faiss = faiss is not None and (vector_db_dir / "law_articles.faiss").exists()
    index = faiss.read_index(str(vector_db_dir / "law_articles.faiss")) if use_faiss else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(detector_path))
    emb_model = CLIPModel.from_pretrained(embedding_model_dir).to(device).eval()
    emb_processor = AutoProcessor.from_pretrained(embedding_model_dir)

    tok = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    outputs = []
    for row in data:
        image_path = image_path_from_id(image_root, row["image_id"])
        crop = _detect_crop(detector, image_path)
        if crop is None:
            crop = Image.open(image_path).convert("RGB")

        qvec = _embed_query(emb_model, emb_processor, device, crop, row["question"])
        if index is not None:
            _, idxs = index.search(qvec, topk)
            top_indices = idxs[0].tolist()
        else:
            scores = qvec @ embeddings.T
            top_indices = np.argsort(-scores[0])[:topk].tolist()
        ctx = [meta[i] for i in top_indices if i >= 0]

        prompt = _build_prompt(row["question"], ctx)
        inputs = tok(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            gen = llm.generate(**inputs, max_new_tokens=16, do_sample=False)
        ans = tok.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

        outputs.append(
            {
                "id": row["id"],
                "image_id": row["image_id"],
                "question": row["question"],
                "relevant_articles": [
                    {"law_id": c["law_id"], "article_id": c["article_id"]} for c in ctx
                ],
                "answer": ans,
            }
        )

    save_json(out_json, outputs)
    print(f"Saved RAG outputs to: {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase 4 RAG QA on public/private split.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["public", "private"], default="public")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--embedding-model-dir", type=Path, required=True)
    parser.add_argument("--vector-db-dir", type=Path, required=True)
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-json", type=Path, default=Path("new_appoarch/artifacts/phase4/predictions.json"))
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    run_rag(
        root=args.root,
        split=args.split,
        detector_path=args.detector,
        embedding_model_dir=args.embedding_model_dir,
        vector_db_dir=args.vector_db_dir,
        llm_name=args.llm,
        out_json=args.out_json if args.out_json.is_absolute() else (args.root / args.out_json),
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
