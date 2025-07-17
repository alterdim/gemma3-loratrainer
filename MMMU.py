import torch
from transformers import pipeline, AutoProcessor
from PIL import Image
import json
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = "google/gemma-3-4b-it"

SHORT_ANS_FORMAT = """{}
Answer the question using a single word or phrase."""
MULTI_ANS_FORMAT = """{}
Answer with the option's letter from the given choices directly."""

class Gemma3:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.pipeline = pipeline(
            task="image-text-to-text",
            model=MODEL_NAME,
            device=0,
            torch_dtype=torch.bfloat16,
            feature_extractor=self.processor.image_processor,
        )

    def infer(self, text, image, options, multiple_choice=False):
        system_prompt = MULTI_ANS_FORMAT if multiple_choice else SHORT_ANS_FORMAT
        user_prompt = text + "\n" + "A : " + str(options[0]) + "\n B : " + str(options[1]) + "\n C : " + str(options[2]) + "\n D : " + str(options[3]) if multiple_choice else text
        template = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        output = self.pipeline(template, max_new_tokens=150)
        return output[0]["generated_text"][-1]["content"]

categories = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
gemma = Gemma3()
for category in categories:
    dataset = load_dataset("MMMU/MMMU", category, split="test")
    correct = 0
    total = 0

    for entry in tqdm(dataset):
        id_row = entry["id"]
        img_1 = entry["image_1"]
        question = entry["question"]
        answers = entry["options"]
        is_mcq = True if entry["question_type"] == "multiple_choice" else False
        try:
            prediction = gemma.infer(question, img_1, answers, is_mcq)
            with open("results.json", 'a') as out:
                out.write(f'"{id_row}": "{prediction}",\n')
        except Exception as e:
            print(f"⚠️ Error on {id_row}: {e}")
            with open("results.json", 'a') as out:
                out.write(f'"{id_row}": "A",\n')
