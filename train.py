from datasets import load_dataset
from PIL import Image
import wandb
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
ROW_NUMBER = 50000 #use none for full set
USE_4BIT_QUANTIZATION = True
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACC_STEPS = 2

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
        "batch_size": 2,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 16,
    }
)
# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["conversations"][1]["value"]}],
            },
        ],
    }

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = Image.open(element["image"]).convert("RGB")
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs



# System message for the assistant
system_message = "You are an expert Python developer specializing in matplotlib."

# User prompt that combines the user query and the schema
user_prompt = """I need your help to generate the Python code that can reproduce the picture based on the picture I provided.
To ensure accuracy and detail in your recreation, you need to begin with a comprehensive analysis of the figure.
You should generate code snippets with the following steps
1.Layout and Chart Type Analysis: e.g., identify the picture's composition, noting the presence, arrangement of any subplots and how many charts are within a subplot.
2.Data Analysis: e.g., summarize the data trend or patter. 
3.Additional Features: e.g., identify any supplementary elements such as legends, colormaps, tick labels, or text annotations that contribute to the figure's clarity or aesthetic appeal.
4.Then generate the final code according to the previous analysis.
"""

dataset = load_dataset("json", data_files="chart2code_160k.json", split="train")
dataset = [format_data(sample) for sample in dataset]

if ROW_NUMBER is not None:
    reduced_dataset = dataset[:ROW_NUMBER]
else:
    reduced_dataset = dataset

MODEL_OUT_NAME = BASE_MODEL + "_chart2code_" + str(len(reduced_dataset))



# Hugging Face model id
model_id = BASE_MODEL # 

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
    device_map="auto", # What device to use, defaults to auto
)

if USE_4BIT_QUANTIZATION:
    # BitsAndBytesConfig int-4 config
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
model.config.use_cache = False
processor = AutoProcessor.from_pretrained(BASE_MODEL)


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
    ],
)



args = SFTConfig(
    output_dir=RUN_NAME,     # directory to save and repository id
    num_train_epochs=1,                         # number of training epochs
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,              # batch size per device during training
    gradient_accumulation_steps=GRADIENT_ACC_STEPS,            # number of steps before performing a backward/update pass
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="epoch",                      # save checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    max_length=None,
    push_to_hub=True,                           # push model to hub
    report_to="wandb",              
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
args.remove_unused_columns = False # important for collator

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch



trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=reduced_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

wandb.finish()

del model
del trainer
torch.cuda.empty_cache()



# Load Model base model
model = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(MODEL_OUT_NAME, safe_serialization=True, max_shard_size="4GB")

processor.save_pretrained(MODEL_OUT_NAME)