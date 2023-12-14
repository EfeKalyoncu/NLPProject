from transformers import AutoTokenizer
import torch


class EventSentenceLoader:
    def __init__(self, filepath, tokenizer_name):
        self.filepath = filepath
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load_data(self):
        data = []
        with open(self.filepath, "r", encoding="utf-8") as file:
            for line in file:
                sentence, event = line.strip().split(" /// ")

                tokenized_sentence = self.tokenizer.tokenize(sentence)
                encoding = self.tokenizer(event, padding='max_length', max_length=256)
                event_tokens = set(self.tokenizer.tokenize(event))

                # Label the sentence tokens
                labels = [
                    1 if token in event_tokens else 0 for token in tokenized_sentence
                ]

                labels += [0] * (256 - len(labels))

                # Append the tokenized sentence and labels to the data list
                data.append(
                    {
                        "sentence": sentence,
                        "tokens": torch.tensor(encoding['input_ids']),
                        "attention": torch.tensor(encoding['attention_mask']),
                        "labels": torch.tensor(labels),
                    }
                )
        return data


# Usage
filepath = "events.txt"
tokenizer_name = "bert-base-cased"
loader = EventSentenceLoader(filepath, tokenizer_name)
data = loader.load_data()
