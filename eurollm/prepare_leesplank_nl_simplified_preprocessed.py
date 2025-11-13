from datasets import load_dataset, interleave_datasets, DatasetDict
import random

SIMPLIFICATION_PROMPTS = [
    "Vereenvoudig deze tekst:",
    "Kun je deze tekst makkelijker maken?",
    "Herschrijf dit in eenvoudigere taal:",
    "Maak deze tekst begrijpelijker:",
    "Schrijf dit in simpelere woorden:",
    "Vertaal dit naar makkelijkere taal:",
    "Maak dit leesbaar voor een breder publiek:",
    "Herschrijf dit zodat iedereen het kan begrijpen:",
    "Maak deze tekst toegankelijker:",
    "Schrijf dit in klare taal:",
    "Vereenvoudig dit stuk tekst:",
    "Kun je dit makkelijker uitleggen?",
    "Maak dit stuk tekst eenvoudiger:",
    "Herschrijf dit in begrijpelijke taal:",
    "Zet dit om naar eenvoudige taal:",
    "Maak dit makkelijker te lezen:",
    "Schrijf dit in simpele taal:",
    "Vertaal dit naar begrijpelijke woorden:",
    "Maak dit tekststuk eenvoudiger:",
    "Herschrijf dit in alledaagse taal:"
]

def format_chat_leesplank(examples, seed=42):
    random.seed(seed)
    formatted_messages = []
    instructions = []

    for prompt, result in zip(examples["prompt"], examples["result"]):
        instruction = "Vereenvoudig: " if (random.random() < 0.2 or len(prompt) < 100) else random.choice(SIMPLIFICATION_PROMPTS)
        formatted_messages.append([
            {"role": "user", "content": f"{instruction} {prompt}"},
            {"role": "assistant", "content": result},
        ])
        instructions.append(instruction)

    return {
        "messages": formatted_messages,
        "instruction": instructions
    }

def main(seed=42, ordered_ratio=0.7):
    # Load all splits
    splits = ["train", "val", "test"]
    datasets = {
        split: load_dataset(
            "UWV/Leesplank_NL_wikipedia_simplifications_preprocessed",
            split=split
        ) for split in splits
    }

    # Randomly choose which examples go in ordered vs shuffled groups
    n = len(datasets["train"])
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # Split indices into ordered and shuffled sets
    cutoff = int(n * ordered_ratio)
    ordered_indices = sorted(indices[:cutoff])  # preserve ascending order
    shuffled_indices = indices[cutoff:]         # will be randomized

    # Create two subsets
    ds_ordered = datasets["train"].select(ordered_indices)
    ds_shuffled = datasets["train"].select(shuffled_indices).shuffle(seed=seed)

    # Interleave with probability-based sampling
    processed_datasets = {
        "train": interleave_datasets(
            [ds_ordered, ds_shuffled],
            probabilities=[ordered_ratio, 1 - ordered_ratio],
            seed=seed,
            stopping_strategy="all_exhausted"
        ),
        "val": datasets["val"].shuffle(seed=seed),
        "test": datasets["test"].shuffle(seed=seed)
    }

    # Format all splits to chatml
    formatted_datasets = {
        split: dataset.map(
            lambda x: format_chat_leesplank(x, seed),
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=None,
        ) for split, dataset in processed_datasets.items()
    }

    # Convert to DatasetDict and push all splits at once
    dataset_dict = DatasetDict(formatted_datasets)
    dataset_dict.push_to_hub(
        "yhavinga/Leesplank_NL_wikipedia_simplifications_preprocessed_chatml_format_2",
        private=True,
        commit_message="Add ChatML formatted Dutch text simplification dataset"
    )

    # Print statistics
    for split, dataset in dataset_dict.items():
        print(f"{split} split size: {len(dataset)}")

    print("\nExample conversation from validation set:")
    print(dataset_dict["val"][0]["messages"])

    return dataset_dict

if __name__ == "__main__":
    main(ordered_ratio=0.7)