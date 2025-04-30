# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from transformers import pipeline

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

master_df = pd.concat(dataframes, ignore_index=True)

spanish_utts = master_df[master_df["UtteranceLang"] == "spanish/catalan"]

third_500 = spanish_utts.iloc[1000:1500]

hf_token = "" # Add your Hugging Face token here

pipe = pipeline(
    "translation", model="facebook/nllb-200-1.3B", token=hf_token, device="cpu"
)

with open("translated_3.txt", "a") as file:
    for i, utt in enumerate(third_500["Utterance"]):
        translated = pipe(utt, src_lang="spa_Latn", tgt_lang="eng_Latn")[0][
            "translation_text"
        ]
        file.write(
            f"""{i + 1}.\n
        Original: {utt}\n
        Translated: {translated}\n"""
        )
