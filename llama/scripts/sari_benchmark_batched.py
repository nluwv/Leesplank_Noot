import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline
from datasets import load_dataset
import torch
from evaluate import load
import pandas as pd
import time

sari = load("sari")

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_AUTH_TOKEN")

login(HUGGINGFACE_TOKEN)

ds = load_dataset("UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split="test")
df = ds.to_pandas()
df_size = len(df)

df_e = df[:int(df_size*(1/3))]
df_m = df[int(df_size*(1/3)):int(df_size*(2/3))].reset_index(drop=True)
df_h = df[:int(df_size*(2/3))].reset_index(drop=True)

datasets = [df_e, df_m, df_h]
system_prompt = "Vereenvoudig een Nederlandse alinea tot één duidelijke en aantrekkelijke tekst geschikt voor volwassenen die Nederlands als tweede taal spreken. Gebruik woorden uit de basiswoordenlijst Amsterdamse kleuters. Behoud directe citaten en leg culturele verwijzingen, uitdrukkingen en technische termen natuurlijk uit in de tekst. Pas de volgorde van informatie aan voor eenvoud en leesbaarheid."

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    padding="longest",
    batch_size=64,
)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]

score_list_e = []
score_list_m = []
score_list_h = []
score_list_total = []

file_path = "outputs/scores/sari_scores_output_batched_padded.txt"

print("start")
start_time = time.time()
for i, data in enumerate(datasets):
    with open(file_path, "a") as file:
        file.write(f"""begin: {i}""" + "\n")
    list_messages = []
    for index, row in data.iterrows():
        prompt = row["prompt"]
        query = f"""{system_prompt} De alinea: {prompt}"""
        messages = [
            {"role": "user", "content": query},        
        ]
        list_messages.append(messages)    
    for index, output in enumerate(pipe(list_messages, add_special_tokens=False, pad_token_id=pipe.tokenizer.eos_token_id)):
        response = output[0]["generated_text"][-1]["content"]
        row = data.loc[index]
        sources = [row["prompt"]]
        predictions = [response]
        references = [[row["result"]]]
        sari_score = sari.compute(sources=sources, predictions=predictions, references=references)["sari"]
        
        with open(file_path, "a") as file:
            file.write(str(sari_score) + '\n')

        if i == 0:
            score_list_e.append(sari_score)
        elif i == 1:
            score_list_m.append(sari_score)
        elif i == 2:
            score_list_h.append(sari_score)
        score_list_total.append(sari_score)
    print("part of dataset done")
print("done")
end_time = time.time()
execution_time = end_time - start_time

score_e = sum(score_list_e)/len(score_list_e)
score_m = sum(score_list_m)/len(score_list_m)
score_h = sum(score_list_h)/len(score_list_h)
score_total = sum(score_list_total)/len(score_list_total)

with open(file_path, "a") as file:
    file.write(f"""Easy sari score: {score_e}""" + '\n')

with open(file_path, "a") as file:
    file.write(f"""Medium sari score: {score_m}""" + '\n')

with open(file_path, "a") as file:
    file.write(f"""Hard sari score: {score_h}""" + '\n')

with open(file_path, "a") as file:
    file.write(f"""Total sari score: {score_total}""" + '\n')

with open(file_path, "a") as file:
    file.write(f"""Execution time: {execution_time}""")

print(f"""Easy category sari score: {score_e}""")
print(f"""Medium category sari score: {score_m}""")
print(f"""Hard category sari score: {score_h}""")
print(f"""Total sari score: {score_total}""")
print(f"""Execution time: {execution_time}""")
