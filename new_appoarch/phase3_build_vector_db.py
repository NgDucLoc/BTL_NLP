from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

from config import get_paths
from utils import build_article_map, load_json, save_json

if sys.version_info < (3, 13):
    import faiss  # type: ignore
else:
    faiss = None


def _to_feature_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported CLIP output type: {type(output)}")


def _embed_texts(texts: list[str], model: CLIPModel, processor: AutoProcessor, device: torch.device) -> np.ndarray:
    vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _to_feature_tensor(model.get_text_features(**inputs))
            out = F.normalize(out, dim=-1)
        vecs.append(out.cpu().numpy())
    return np.concatenate(vecs, axis=0)


def build_vector_db(root: Path, embedding_model_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = get_paths(root)

    law_db = load_json(paths.law_json)
    article_map = build_article_map(law_db)
    meta = []
    texts = []
    for (law_id, article_id), text in article_map.items():
        meta.append({"law_id": law_id, "article_id": article_id, "text": text})
        texts.append(text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(embedding_model_dir).to(device).eval()
    processor = AutoProcessor.from_pretrained(embedding_model_dir)

    embeddings = _embed_texts(texts, model, processor, device).astype("float32")
    if faiss is not None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "law_articles.faiss"))
    else:
        print("FAISS disabled for this Python version; falling back to numpy retrieval in phase 4.")
    np.save(out_dir / "law_articles.npy", embeddings)
    save_json(out_dir / "law_articles_meta.json", meta)
    print(f"Vector DB built at: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector DB from finetuned embeddings.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--embedding-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("new_appoarch/artifacts/phase3"))
    args = parser.parse_args()

    build_vector_db(
        root=args.root,
        embedding_model_dir=args.embedding_model_dir,
        out_dir=args.out_dir if args.out_dir.is_absolute() else (args.root / args.out_dir),
    )


if __name__ == "__main__":
    main()
