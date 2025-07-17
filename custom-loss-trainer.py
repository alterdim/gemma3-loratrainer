import os
import json
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from span_func import find_important_spans
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# === Setup ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_ID = "google/gemma-3-4b-it"
DATA_PATH = "chart2code_160k.json"
ROW_NUMBER = 1000
SEMANTIC_TOKEN_MULT = 3
MAX_LENGTH = 4096

OUTPUT_DIR = "gemma-chart2code-lora-customhead"

# Token weighting options
USE_PLT = True
USE_AX = True
USE_ASSIGNMENTS = True
USE_TITLE = True
USE_KWARGS = True
USE_TRIPLE_STRINGS = True

# === Dataset ===
class Chart2CodeDataset(Dataset):
    def __init__(self, json_path, row_number=None):
        with open(json_path, "r") as f:
            raw_samples = json.load(f)

        if row_number:
            raw_samples = raw_samples[:row_number]

        self.samples = []
        self.instruction = (
            "You are an expert developer specializing in writing Python matplotlib code based on a given picture. "
            "I need your help to generate the Python code that can reproduce the picture based on the picture I provided.\n"
            "To ensure accuracy and detail in your recreation, you need to begin with a comprehensive analysis of the figure.\n"
            "You should generate code snippets with the following steps:\n"
            "1. Layout and Chart Type Analysis\n"
            "2. Data Analysis\n"
            "3. Additional Features\n"
            "4. Then generate the final code according to the previous analysis."
        )

        for sample in tqdm(raw_samples):
            image_path = sample.get("image")
            if not image_path or not os.path.exists(image_path):
                continue
            try:
                with Image.open(image_path) as im:
                    im.verify()
                self.samples.append(sample)
            except (UnidentifiedImageError, OSError):
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image"]).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self))

        assistant_code = sample["conversations"][1]["value"]
        spans = find_important_spans(
            assistant_code,
            USE_PLT,
            USE_AX,
            USE_ASSIGNMENTS,
            USE_TITLE,
            USE_KWARGS,
            USE_TRIPLE_STRINGS
        )
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.instruction},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_code}],
                },
            ],
            "important_code": assistant_code,
            "important_spans": spans,
        }

# === Data prep ===
dataset = Chart2CodeDataset(DATA_PATH, ROW_NUMBER)

# === Model & Processor ===
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    quantization_config=quant_config,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# === Collate function ===
def collate_fn(examples):
    texts = []
    images = []
    all_spans = []
    important_codes = []

    for example in examples:
        msg = example["messages"]
        text = processor.apply_chat_template(msg, add_generation_prompt=False, tokenize=False).strip()
        texts.append(text)

        image_inputs = []
        for m in msg:
            for c in m["content"]:
                if c.get("type") == "image":
                    image_inputs.append(c["image"].convert("RGB"))
        images.append(image_inputs)

        all_spans.append(example["important_spans"])
        important_codes.append(example["important_code"])

    # Tokenize full message (chat template) for loss weighting
    tokenized = processor.tokenizer(
        texts,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    offset_mappings = tokenized["offset_mapping"]

    loss_weights_batch = []
    for offsets, spans in zip(offset_mappings, all_spans):
        weights = [1.0] * len(offsets)
        for i, (start, end) in enumerate(offsets):
            for span_start, span_end in spans:
                if start >= span_start and end <= span_end:
                    weights[i] = SEMANTIC_TOKEN_MULT
        loss_weights_batch.append(torch.tensor(weights, dtype=torch.float32))

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100  # fallback image token ID

    batch["labels"] = labels

    loss_weights = pad_sequence(loss_weights_batch, batch_first=True, padding_value=1.0)

    # Match lengths
    min_len = min(loss_weights.shape[1], labels.shape[1])
    loss_weights = loss_weights[:, :min_len]
    labels = labels[:, :min_len]
    batch["input_ids"] = batch["input_ids"][:, :min_len]
    batch["attention_mask"] = batch["attention_mask"][:, :min_len]
    batch["labels"] = labels
    batch["loss_weights"] = loss_weights

    return batch

# === LoRA Config ===
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    modules_to_save=["lm_head"],
)

# === Custom Trainer ===
class WeightedLossTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        import torch.nn.functional as F
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        )

        if loss_weights is not None:
            flat_weights = loss_weights.view(-1)
            if flat_weights.shape[0] != loss.shape[0]:
                min_len = min(flat_weights.shape[0], loss.shape[0])
                flat_weights = flat_weights[:min_len]
                loss = loss[:min_len]
            loss = loss * flat_weights

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

# === Training config ===
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    push_to_hub=False,
    report_to="wandb",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
sft_args.remove_unused_columns = False

# === Trainer ===
trainer = WeightedLossTrainer(
    model=model,
    args=sft_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

trainer.train()

model = PeftModel.from_pretrained(model, trainer.peft_config[OUTPUT_DIR])
model = model.merge_and_unload()
# Save model and processor
model.save_pretrained("gemma-chart2code-lora-merged")
processor.save_pretrained("gemma-chart2code-lora-merged")