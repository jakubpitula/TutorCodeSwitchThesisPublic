import pandas as pd
from pathlib import Path
import random
import torch

data_path = Path("./data/processed/new/BELC")
dataframes = []

# We're only interested in the interview data

dir_name = "2-interview_10-16"
sub_dirs = [Path(data_path / dir_name)]
while sub_dirs:
    current_dir = sub_dirs.pop()
    for sub_dir in current_dir.iterdir():
        if sub_dir.is_dir():
            sub_dirs.append(sub_dir)
        else:
            if sub_dir.suffix == ".pkl":
                # Sub dir is a pickle file
                df = pd.read_pickle(sub_dir)
                dataframes.append(df)

# Get proficiency group names from the data

proficiency_group_names = set()
for df in dataframes:
    for group_name, group_df in df.groupby("Proficiency"):
        proficiency_group_names.add(group_name)

# Split dataframes by proficiency group

proficiency_dataframes = {str(group_name): [] for group_name in proficiency_group_names}
for df in dataframes:
    for group_name, group_df in df.groupby("Proficiency"):
        proficiency_dataframes[str(group_name)].append(group_df)

for group_name, group_dataframes in proficiency_dataframes.items():
    print(f"Proficiency group {group_name}: {len(group_dataframes)}")

# Randomly select 10% of the dialogues for each group

random.seed(42)  # for results replicability

selected_group_dataframes = {}
for group in proficiency_dataframes.keys():
    group_size = len(proficiency_dataframes[group])
    sample_size = max(1, group_size // 10)  # Ensure at least one sample
    selected_group_indices = random.sample(range(group_size), sample_size)
    print(f"Selected {sample_size} samples from group {group} of size {group_size}")

    selected_group_dataframes[group] = [
        proficiency_dataframes[group][i] for i in selected_group_indices
    ]

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

hf_token = ""  # Add your Hugging Face token here

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", token=hf_token
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", token=hf_token, low_cpu_mem_usage=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

for group_name, group_dataframes in selected_group_dataframes.items():
    for df in group_dataframes:
        utterances = df["Utterance"].where(df["Participant"] == "PAR").dropna()
        for utterance in utterances[:1]:
            messages = [
                {
                    "role": "system",
                    "content": """
              You will be given an english learner's utterance in a dialogue with a teacher.\n
                  Your task is to decide, whether this utterance shows signs of non understanding.\n
                  Answer simply 'yes' or 'no'. Do not engage in a conversation with the user.\n
                  Just tell me, whether their utterance signals non understading.\n
                  Your answer should ONLY contain a single word. yes or no.\n
                  Only answer 'yes' if the utterance shows signs of non understanding. If it doesn't - answer 'no'.
                  Do not complete the sentence or utterance.
                  Do not engage in the conversation.
                  Simply answer 'yes' or 'no'.
              """,
                },
                {"role": "user", "content": f"{utterance}"},
            ]
            res = pipe(
                messages,
                return_full_text=False,
                max_new_tokens=20,
            )
            print(f"Utterance: {utterance}\nDecision: {res[0]['generated_text']}")
            with open("intent_labels.txt", "w") as file:
                file.write(f"\nUTTERANCE: {utterance}\n")

                for seq in res:
                    file.write(f"LABEL: {seq['generated_text']}\n")

                file.write("\n==================================\n")
