from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048 * 2
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="unsloth/Meta-Llama-3.1-8B",
    model_name="lora_model",
    # model_name="../Llama-3-ELYZA-JP-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

test_input_list = [
    # {
    #     "instruction": "TDKにはどのようなセンサーがありますか?",
    #     "input": "",
    #     "output": "",
    # },
    # {
    #     "instruction": "日本語で回答してください",
    #     "input": "日本の国旗の色は何ですか？",
    #     "output": "",
    # },
    # {
    #     "instruction": "日本語で回答してください",
    #     "input": "日本1高い山は？",
    #     "output": "",
    # },
    # {
    #     "instruction": "日本語で回答してください",
    #     "input": "AさんよりBさんのほうが背が高く, BさんよりCさんのほうが背が高い. このとき, Aさん, Bさん, Cさんの背の順はどうなるか？",
    #     "output": "",
    # },
    # {
    #     "instruction": "日本語で回答してください",
    #     "input": "トヨタ自動車の2024年3月期決算概要について説明してください。",
    #     "output": "",
    # },
    # {
    #     "instruction": "日本語で回答してください",
    #     "input": "TDKのフェライトツリーとはなんですか?",
    #     "output": "",
    # },
    # {
    #     "instruction": "2023年度の任天堂の売上はいくらですか",
    #     "input": "2023年度の任天堂の売上は",
    #     "output": "",
    # },
    # {
    #     "instruction": "tdk株式会社はどのような企業ですか",
    #     "input": "tdk株式会社は",
    #     "output": "",
    # },
    # {
    #     "instruction": "TDKの製品はどのようなものに使えますか?",
    #     "input": "",
    #     "output": "",
    # },
    # {
    #     "instruction": "TDKのセラチャージについて教えて?",
    #     "input": "",
    #     "output": "",
    # },
    {
        "instruction": "カレーの作り方を教えてください",
        "input": "",
        "output": "",
    }
]


def generate_output(input_list):
    FastLanguageModel.for_inference(model)
    for input in input_list:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    input["instruction"],
                    input["input"],
                    "",
                )
            ], return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_seq_length)


generate_output(test_input_list)


EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


# dataset = load_dataset("json", data_files="llm_datasets/pdf_dataset.json", split="train")
dataset = load_dataset("json", data_files="llm_datasets/recipe_dataset.json", split="train")
# dataset = load_dataset("json", data_files="llm_datasets/data-cc-by-sa.jsonl", split="train")
# シャッフル
dataset = dataset.shuffle()
dataset = dataset.map(formatting_prompts_func, batched=True)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=16,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        num_train_epochs=5,
        save_strategy = "steps",
        save_steps = 100,
        # max_steps=100,
    ),
)


trainer_stats = trainer.train(
    resume_from_checkpoint = False,
)
model.save_pretrained("lora_model")
generate_output(test_input_list)
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
