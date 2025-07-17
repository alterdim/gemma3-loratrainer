import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from unsloth.utils import patch_logits_output
patch_logits_output(True)
from unsloth import FastVisionModel # FastLanguageModel for LLMs
from modelscope.msdatasets import MsDataset
import json
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import wandb
from torch.utils.data import Dataset
from unsloth import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedTokenizer
import torch


BASE_MODEL = "google/gemma-3-4b-it-unsloth"
ROW_NUMBER = 1000
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACC_STEPS = 4
SEMANTIC_TOKEN_MULT = 3
MAX_LENGTH = 4096

if ROW_NUMBER is not None:
    RUN_NAME = BASE_MODEL + "-" + str(ROW_NUMBER)
else:
    RUN_NAME = BASE_MODEL + "-full"

RUN_NAME = RUN_NAME + "-customloss"



class WeightedLossTrainer(SFTTrainer): 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): #kwargs necessary to avoid extra arguments being passed
        loss_weights = inputs.pop("loss_weights", None)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        ## LOSS OVERRIDE
        import torch.nn.functional as F
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        )

        if loss_weights is not None:
            ## flatten
            loss = loss * loss_weights.view(-1)

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3-4b-it",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
    ],
)

class WeightedVisionDataCollator(UnslothVisionDataCollator):
    def __call__(self, instances):
        batch = super().__call__(instances)
        tokenizer = self.processor.tokenizer

        loss_weights_batch = []
        for instance in instances:
            code = instance["important_code"]
            spans = instance["important_spans"]

            tokenized = tokenizer(code, return_offsets_mapping=True, truncation=True, max_length=MAX_LENGTH)
            weights = [1.0] * len(tokenized["input_ids"])

            for i, (start, end) in enumerate(tokenized["offset_mapping"]):
                for span_start, span_end in spans:
                    if start >= span_start and end <= span_end:
                        weights[i] = SEMANTIC_TOKEN_MULT  #MAGIC

            weights += [1.0] * (MAX_LENGTH - len(weights))
            weights = weights[:MAX_LENGTH]

            loss_weights_batch.append(weights)

        batch["loss_weights"] = torch.tensor(loss_weights_batch, dtype=torch.float32)
        return batch

class Chart2CodeDataset(Dataset):
    def __init__(self, json_path, row_number=None):
        with open(json_path, "r") as f:
            raw_samples = json.load(f)

        if row_number is not None:
            raw_samples = raw_samples[:row_number]

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

        self.samples = []
        for sample in tqdm(raw_samples):
            image_path = sample.get("image")
            if not image_path or not os.path.exists(image_path):
                continue
            try:
                with Image.open(image_path) as im:
                    im.verify()  # Vérifie la validité du fichier image
                self.samples.append(sample)
            except (UnidentifiedImageError, OSError):
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image"]).convert("RGB")
        except Exception as e:
        
            return self.__getitem__((idx + 1) % len(self))
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["conversations"][1]["value"]}],
            },
        ]
        assistant_code = sample["conversations"][1]["value"]
        important_spans = find_important_spans(assistant_code)
        return {
            "messages": conversation,
            "important_code": assistant_code,
            "important_spans": important_spans,
        }

converted_dataset = Chart2CodeDataset("chart2code_160k.json", ROW_NUMBER)



def find_important_spans(code: str):
    import re
    spans = []
    for match in re.finditer(r"plt\.(plot|bar|scatter)\((.*?)\)", code):
        spans.append(match.span(2))
    for match in re.finditer(r"plt\.(title|xlabel|ylabel)\((.*?)\)", code):
        spans.append(match.span(2))
    return spans


processor = get_chat_template(
    processor,
    "gemma-3"
)


FastVisionModel.for_training(model) 

wandb.init(
    project="gemma-chart2code-lora-newloss",
    name=RUN_NAME,
    config={
        "model": BASE_MODEL,
        "dataset": "chart2code_160k",
        "epochs": 1,
        "batch_size": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACC_STEPS,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 16,
    }
)

trainer = WeightedLossTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=WeightedVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACC_STEPS,
        gradient_checkpointing = True,

        # use reentrant checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,        
        warmup_ratio = 0.03,
        num_train_epochs = 1,         
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy="steps",
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", 

        #VISION STUFF
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_seq_length = MAX_LENGTH,
    )
)

trainer_stats = trainer.train()

model.save_pretrained_merged(RUN_NAME, processor,)