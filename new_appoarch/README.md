# New Multimodal 4-Phase Pipeline

Pipeline nay duoc thiet ke theo dung flow ban de xuat:

1. Phase 1: augment du lieu bien bao + du lieu thuc te, train YOLO detector vi tri bien bao.
2. Phase 2: dung detector de crop bien bao tren train set, finetune multimodal embedding tren quan he question-article.
3. Phase 3: dua article embedding vao vector DB (FAISS) de retrieval.
4. Phase 4: ket hop ket qua retrieval vao LLM de tra loi cau hoi.

## Cau truc file

- `phase1_generate_detection_dataset.py`: Tao synthetic YOLO dataset tu anh train thuc te + icon bien bao trong law_db/images.fld.
- `phase1_train_yolo.py`: Train detector YOLO.
- `phase2_build_multimodal_samples.py`: Chay detector tren train image, crop bien bao, tao cap positive/negative article.
- `phase2_finetune_embedding.py`: Finetune CLIP embedding bang objective contrastive.
- `phase3_build_vector_db.py`: Embed article va build FAISS index.
- `phase4_rag_qa.py`: Retrieval + LLM answering (public/private split).
- `run_pipeline.py`: Chay full end-to-end pipeline.

## Cai dat

Tai root repo:

```bash
pip install -r new_appoarch/requirements.txt
```

## Chay tung phase

### Phase 1: Tao du lieu + train detector

```bash
python new_appoarch/phase1_generate_detection_dataset.py \
  --root /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main \
  --out-dir new_appoarch/artifacts/phase1/yolo_data \
  --synth-per-real 3

python new_appoarch/phase1_train_yolo.py \
  --data-yaml /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase1/yolo_data/traffic_sign.yaml \
  --out-dir /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase1 \
  --epochs 30 --device 0
```

### Phase 2: Tao sample + finetune embedding

```bash
python new_appoarch/phase2_build_multimodal_samples.py \
  --root /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main \
  --detector /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase1/yolo_sign_detector/weights/best.pt \
  --out-json /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase2/multimodal_train_samples.json

python new_appoarch/phase2_finetune_embedding.py \
  --train-samples /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase2/multimodal_train_samples.json \
  --output-dir /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase2/clip_finetuned \
  --epochs 2
```

### Phase 3: Build vector DB

```bash
python new_appoarch/phase3_build_vector_db.py \
  --root /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main \
  --embedding-model-dir /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase2/clip_finetuned \
  --out-dir /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase3
```

### Phase 4: Retrieval + LLM QA

```bash
python new_appoarch/phase4_rag_qa.py \
  --root /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main \
  --split public \
  --detector /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase1/yolo_sign_detector/weights/best.pt \
  --embedding-model-dir /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase2/clip_finetuned \
  --vector-db-dir /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase3 \
  --llm Qwen/Qwen2.5-3B-Instruct \
  --out-json /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main/new_appoarch/artifacts/phase4/public_predictions.json
```

## Chay full pipeline

```bash
python new_appoarch/run_pipeline.py \
  --root /Users/loclinh/Downloads/VLSP2025-MLQA-TSR-main \
  --device 0 \
  --llm Qwen/Qwen2.5-3B-Instruct
```

## Ghi chu quan trong

- Dataset goc khong co bbox ground truth cho bien bao, vi vay phase 1 dang dung huong synthetic-supervision de bootstrap detector.
- De nang chat luong detector, ban nen bo sung mot tap nho annotation that (manual label) va retrain/fine-tune YOLO.
- Co the thay model LLM bang model tieng Viet phu hop hon neu ban co GPU VRAM lon hon.
