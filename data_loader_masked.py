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
                sentence, event, new_sentence = line.strip().split(" /// ")
                maskable = torch.zeros((1, 256))

                tokenized_sentence = self.tokenizer.tokenize(sentence)
                tokenized_new_sentence = self.tokenizer.tokenize(new_sentence)
                tokenized_event = self.tokenizer.tokenize(event)
                for i in range(len(tokenized_sentence) + len(tokenized_new_sentence) - len(tokenized_event) + 1, len(tokenized_sentence) + len(tokenized_new_sentence) + 1):
                    maskable[0, i] = 1
                new_sentence = sentence + " " + new_sentence
                encoding = self.tokenizer(new_sentence, padding='max_length', max_length=256)
                event_tokens = set(self.tokenizer.tokenize(event))

                # Append the tokenized sentence and labels to the data list
                data.append(
                    {
                        "sentence": sentence,
                        "new_sentence": new_sentence,
                        "tokens": torch.tensor(encoding['input_ids']),
                        "maskable": maskable,
                        "attention": torch.Tensor(encoding['attention_mask'])
                    }
                )
        return data


# Usage
filepath = "new_sentences.txt"
tokenizer_name = "roberta-base"
loader = EventSentenceLoader(filepath, tokenizer_name)
data = loader.load_data()
