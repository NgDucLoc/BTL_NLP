from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from utils import load_json


def _to_feature_tensor(output):
    # transformers versions may return a tensor or a model output object.
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


class MultimodalTripletDataset(Dataset):
    def __init__(self, data_path: Path):
        self.samples = load_json(data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "crop_path": s["crop_path"],
            "question": s["question"],
            "pos_text": s["positive"]["text"],
            "neg_text": s["negative"]["text"],
        }


def _collate(batch, processor):
    images = [Image.open(b["crop_path"]).convert("RGB") for b in batch]
    image_texts = [b["question"] for b in batch]
    pos_texts = [b["pos_text"] for b in batch]
    neg_texts = [b["neg_text"] for b in batch]

    image_inputs = processor(images=images, text=image_texts, return_tensors="pt", padding=True, truncation=True)
    pos_inputs = processor(text=pos_texts, return_tensors="pt", padding=True, truncation=True)
    neg_inputs = processor(text=neg_texts, return_tensors="pt", padding=True, truncation=True)
    return image_inputs, pos_inputs, neg_inputs


def finetune(
    train_samples: Path,
    output_dir: Path,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 8,
    epochs: int = 2,
    lr: float = 2e-5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    ds = MultimodalTripletDataset(train_samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: _collate(b, processor))

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{epochs}")
        for image_inputs, pos_inputs, neg_inputs in pbar:
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
            neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}

            out_im = _to_feature_tensor(model.get_image_features(pixel_values=image_inputs["pixel_values"]))
            out_q = _to_feature_tensor(model.get_text_features(
                input_ids=image_inputs["input_ids"],
                attention_mask=image_inputs["attention_mask"],
            ))
            out_pos = _to_feature_tensor(model.get_text_features(
                input_ids=pos_inputs["input_ids"],
                attention_mask=pos_inputs["attention_mask"],
            ))
            out_neg = _to_feature_tensor(model.get_text_features(
                input_ids=neg_inputs["input_ids"],
                attention_mask=neg_inputs["attention_mask"],
            ))

            query_embed = F.normalize(out_im + out_q, dim=-1)
            pos_embed = F.normalize(out_pos, dim=-1)
            neg_embed = F.normalize(out_neg, dim=-1)

            pos_sim = (query_embed * pos_embed).sum(dim=-1)
            neg_sim = (query_embed * neg_embed).sum(dim=-1)
            loss = F.relu(0.2 - pos_sim + neg_sim).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved finetuned multimodal embedding model to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune multimodal embedding model with detector crops + question + article semantics.")
    parser.add_argument("--train-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("new_appoarch/artifacts/phase2/clip_finetuned"))
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    finetune(
        train_samples=args.train_samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
