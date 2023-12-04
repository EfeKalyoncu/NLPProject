import xml.etree.ElementTree as ET
from transformers import AutoTokenizer


class ACEEventLoader:
    def __init__(self, xml_file_path, sgm_file_path, tokenizer_name):
        self.xml_file_path = xml_file_path
        self.sgm_file_path = sgm_file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load_sgm_text(self):
        with open(self.sgm_file_path, "r", encoding="utf-8") as file:
            sgm_text = file.read()
        return sgm_text

    def get_event_labels(self, charseq_element, tokenized_sentence):
        start = int(charseq_element.get("START"))
        end = int(charseq_element.get("END"))
        event_labels = [0] * len(tokenized_sentence)

        # Get the character positions of each token
        current_position = 0
        for i, token in enumerate(tokenized_sentence):
            token_start = current_position
            token_end = current_position + len(token)

            # If token falls within event character range, label it as "1"
            if start <= token_start and token_end <= end:
                event_labels[i] = 1

            # Update the current position in the sentence
            current_position += len(token) + 1

        return event_labels

    def tokenize_and_label(self, charseq_element, sgm_text):
        event_text = sgm_text[
            int(charseq_element.get("START")) : int(charseq_element.get("END"))
        ]
        tokenized_sentence = self.tokenizer.tokenize(event_text)
        labels = self.get_event_labels(charseq_element, tokenized_sentence)
        return tokenized_sentence, labels

    def load_and_process_data(self):
        # Parse the XML file
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()

        sgm_text = self.load_sgm_text()

        data = []

        # Iterate over each event in the XML
        for charseq in root.iter("charseq"):
            tokens, labels = self.tokenize_and_label(charseq, sgm_text)
            data.append({"tokens": tokens, "labels": labels})

        return data


# Usage
xml_file_path = "path_to_ace2005_annotation_file.xml"
sgm_file_path = "path_to_corresponding_sgm_file.sgm"
tokenizer_name = "bert-base-uncased"

loader = ACEEventLoader(xml_file_path, sgm_file_path, tokenizer_name)
data = loader.load_and_process_data()
