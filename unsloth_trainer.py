from unsloth import FastVisionModel # FastLanguageModel for LLMs
from modelscope.msdatasets import MsDataset
import json
from tqdm import tqdm
from PIL import Image
import wandb
from torch.utils.data import Dataset
from unsloth import get_chat_template

BASE_MODEL = "google/gemma-3-4b-it-unsloth"
ROW_NUMBER = None
PER_DEVICE_TRAIN_BATCH_SIZE = 32
GRADIENT_ACC_STEPS = 4

if ROW_NUMBER is not None:
    RUN_NAME = BASE_MODEL + "-" + str(ROW_NUMBER)
else:
    RUN_NAME = BASE_MODEL + "-full"

wandb.init(
    project="gemma-chart2code-lora",  # Name your project
    name=RUN_NAME,  # Name this specific run
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

class Chart2CodeDataset(Dataset):
    def __init__(self, json_path, row_number=None):
        with open(json_path, "r") as f:
            self.samples = json.load(f)
        if row_number is not None:
            self.samples = self.samples[:row_number]
        self.instruction = "You are an expert developer specializing in writing Python matplotlib code based on a given picture. I need your help to generate the Python code that can reproduce the picture based on the picture I provided.\nTo ensure accuracy and detail in your recreation, you need to begin with a comprehensive analysis of the figure.    You should generate code snippets with the following steps\n1.Layout and Chart Type Analysis: e.g., identify the picture’s composition, noting the presence,arrangement of any subplots and how many charts are within a subplot.\n2.Data Analysis: e.g., summarize the data trend or pattern.\n3.Additional Features: e.g., identify any supplementary elements such as legends, colormaps, tick labels, or text annotations that contribute to the figure’s clarity or aesthetic appeal.\n4.Then generate the final code according to the previous analysis."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "image", "image": Image.open(sample["image"]).convert("RGB")},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample["conversations"][1]["value"]}]},
        ]
        return {"messages": conversation}

converted_dataset = Chart2CodeDataset("chart2code_160k.json", ROW_NUMBER)



processor = get_chat_template(
    processor,
    "gemma-3"
)

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACC_STEPS,
        gradient_checkpointing = True,

        # use reentrant checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,              # max gradient norm based on QLoRA paper
        warmup_ratio = 0.03,
        num_train_epochs = 1,         # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy="steps",
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",             # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_seq_length = 4096,
    )
)

trainer_stats = trainer.train()

model.save_pretrained_merged(RUN_NAME, processor,)