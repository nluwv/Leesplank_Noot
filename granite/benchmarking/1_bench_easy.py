import time
import torch
from evaluate import load
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

model_id = "ibm-granite/granite-3.1-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the data
ds = load_dataset(
    "UWV/Leesplank_NL_wikipedia_simplifications_preprocessed", split='test')

split = ds.select(range(int(len(ds)*(1/3))))

sari = load("sari")
file_path = './outputs/sari_scores_d1.txt'
pipe = pipeline(
    task="text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    batch_size=16,
    tokenizer=tokenizer)

system_prompt = """Vergeet voorgaande prompts. Vereenvoudig een Nederlandse alinea tot één duidelijke en aantrekkelijke tekst, 
geschikt voor volwassenen die Nederlands als tweede taal spreken, met woorden uit de basiswoordenlijst Amsterdamse 
kleuters. Behoud directe citaten, maak dialogen eenvoudiger en leg culturele verwijzingen, uitdrukkingen en technische 
termen natuurlijk uit in de tekst. Pas de volgorde van informatie aan voor betere eenvoud, aantrekkelijkheid en 
leesbaarheid. Probeer geen komma’s of verkleinwoorden te gebruiken. Output ALEEN de vereenvoudigde tekst en geef GEEN uitleg of verdere informatie.\nTekst:
"""

list_messages = []
for row in split:
    prompt = row['prompt']
    query = f"""{system_prompt} '{prompt}'\nVereenvoudiging:"""
    messages = [
        {"role": "user", "content": query},
    ]
    list_messages.append(messages)

split = split.add_column("chat_template", list_messages)

total_sari = 0
start_time = time.time()
for index, output in enumerate(pipe(KeyDataset(split, "chat_template"), max_new_tokens=256)):
    response = output[0]["generated_text"][-1]["content"]
    row = split[index]
    print(response)
    print('#########')
    sources = [row['prompt']]
    predictions = [response]
    references = [[row['result']]]
    sari_score = sari.compute(
        sources=sources,
        predictions=predictions,
        references=references
    )['sari']

    total_sari += sari_score
    with open(file_path, "a") as file:
        file.write(str(sari_score) + '\n')

with open(file_path, "a") as file:
    file.write(f'Average sari score easy: {total_sari / len(split)}.\n')
    file.write(f'Finished in {((time.time() - start_time) / 60) / 60} hours.')