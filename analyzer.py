from typing import Dict
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Analyzer:
    model_dir: str
    tokenizer: T5Tokenizer
    model: T5ForConditionalGeneration
    character_dict: Dict[int, str]

    def __init__(self, model_dir: str, character_dict: Dict[int, str]) -> None:
        self.model_dir = model_dir
        self.character_dict = character_dict
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir, is_fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        if torch.cuda.is_available():
            self.model.cuda()

    def get_character_from_sentence(self, sentence: str) -> str:
        """文章からキャラクターを推定する"""
        input_ids = self.tokenizer(
            sentence,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).input_ids
        outputs = self.model.generate(input_ids)
        resp = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if int(resp) not in self.character_dict.keys():
            raise ValueError(f"ID:{resp} is not in character_dict")
        return self.character_dict[int(resp)]


if __name__ == "__main__":
    model_dir = "model"
    character_dict = {
        0: "香風智乃",
        1: "保登心愛",
        2: "天々座理世",
        3: "桐間紗路",
        4: "宇治松千夜",
        5: "条河麻耶",
        6: "奈津恵",
    }
    cl = Analyzer(model_dir, character_dict)
    print(cl.get_character_from_sentence("おはようございます。"))
