import argparse
import string
import time
from pathlib import Path

import numpy as np
import torch
import spacy
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics import MeanMetric
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from util.model import OnlineModel
from util.metric import CLIPEvaluator
from util.matcher import IncontextMatcher
from util.util import load_list

PROJECT_ROOT = Path(__file__).resolve().parent

def metric_to_float(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--oracle", type=str,
        choices=["SD3.5", "FLUX", "ShuttleDiffusion"],
        default=None, required=True,
        help="oracle model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["DALLEPrompt"],
        default=None,
        required=True,
        help="directory of data and prompt.",
    )
    parser.add_argument(
        "--max_budget",
        type=int,
        default=200,
        help=(
            "The maximum budget the adversary can use to steal the prompt. The budget is the number of queries to the Oracle."
        ),
    )
    parser.add_argument(
        "--caption_repeat",
        type=int,
        default=400,
        help=(
            "The maximum sample used to generate dynamic modifiers."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images."
        ),
    )
    args = parser.parse_args()

    return args

class LoadDataset(Dataset):
    def __init__(
        self,
        dataset=None,
        oracle=None,
        size=512,
    ):

        self.size = size
        self.oracle = oracle
        self.dataset = dataset
        self.data_dir = PROJECT_ROOT / "data" / "datasets" / self.dataset

        self.image_dir = self.data_dir / "showcase" / self.oracle
        self.prompt_dir = self.data_dir / "prompts.txt"

        with open(self.prompt_dir, "r") as f:
            self.prompts = f.readlines()
        
        self.length = len(self.prompts)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        example = {}
        info = self.prompts[index]
        image_name, prompt = info.split(": ", 1)
        prompt = prompt.strip()

        image_dir = self.image_dir / image_name
        image = Image.open(image_dir).convert('RGB')
        example["image"] = image.resize((self.size, self.size))
        example["prompt"] = prompt
        example["image_name"] = image_name.split(".")[0]

        return example

class Prometheus(torch.nn.Module):
    def __init__(self, caption_repeat=400, query_API=None, evaluator=None, max_budget=100,
                    threshold=0.005, detail_rate=0.2, modifier_rate=0.8):
        super(Prometheus, self).__init__()
        self.caption_repeat = caption_repeat
        self.query_API = query_API
        self.evaluator = evaluator


        # Reserve one query for the base caption generated in caption_base().
        self.detail_budget = max(0, int(max_budget * detail_rate) - 1)
        self.modifier_budget = int(max_budget * modifier_rate)
        self.max_budget = max_budget
        self.threshold = threshold

        self.reset_state()


        from util.BLIP.blip import blip_decoder
        self.caption_model = blip_decoder(
            pretrained=str(PROJECT_ROOT / "data" / "model_base_caption_capfilt_large.pth"),
            med_config=str(PROJECT_ROOT / "util" / "BLIP" / "configs" / "med_config.json"),
            image_size=384,
            vit='base',
        )
        self.caption_model = self.caption_model.to("cuda")
        self.caption_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.spacy_model = spacy.load("en_core_web_trf")

        self.am = IncontextMatcher()

    def reset_state(self):
        self.final_dynamic = []
        self.final_static = []

        self.best_score = 0
        self.base_prompt = ""
        self.final_prompt = ""

        self.detail_list = []
        self.detail_score = []

        self.position_list = []
        self.position_score = []

        self.modifier_list = []
        self.modifier_score = []

    def score_fn(self, lpips_score, ii_score):
        final_score = ii_score + (1 - lpips_score)
        return final_score
        
    def caption_base(self):

        with torch.inference_mode():
            caption = self.caption_model.generate(
                self.showcase_input,
                sample=True,
                max_length=20,
                min_length=5,
            )[0]
        print("The base caption is: ", caption)
        self.base_prompt = caption
        query_result = self.query_API.query(caption)

        lpips_similarity = self.evaluator.LPIPS(self.showcase, query_result)
        ii_similarity = self.evaluator.CLIP_II(self.showcase, query_result)

        self.best_score = self.score_fn(lpips_score=lpips_similarity, ii_score=ii_similarity)

        return

    def obtain_detail_position(self):
        def normalize(text):
            # lower the text
            text = text.lower()
            # remove extra spaces
            text = ' '.join(text.split())
            # remove the punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # remove the stop words
            stop_words = {"a", "an", "the", "of"}
            text = ' '.join([word for word in text.split() if word not in stop_words])
            # convert to base form
            doc = self.spacy_model(text)
            text = ' '.join([token.lemma_ for token in doc])
            return text

        # get detail and position seperately
        for _ in range(self.caption_repeat):
            with torch.inference_mode():
                caption = self.caption_model.generate(
                    self.showcase_input,
                    sample=True,
                    max_length=30,
                    min_length=10,
                )[0]

            print("The sampled caption is: ", caption)
            doc = self.spacy_model(caption)

            # get detailed subjects
            for noun in doc.noun_chunks:
                # doing normalization
                text = normalize(noun.text)
                # limit the length of the text to avoid overfitting
                if text and text not in self.detail_list and len(text.split()) <= 3:
                    self.detail_list.append(text)

            # get the position
            position_prepositions = {
                "in", "on", "at", "by", "near", "under", "over", "between", "through",
                "inside", "outside", "beside", "beneath", "above", "below", "within",
                "among", "around", "across", "along", "behind", "beyond", "toward",
                "towards", "onto", "upon", "against", "amid", "amidst"
                }
            for token in doc:
                if token.dep_ == "prep" and token.text.lower() in position_prepositions: # only consider the prepositions related to the position in the list
                    # ensure the preposition has an pobj
                    has_object = any(child.dep_ == "pobj" for child in token.children)
                    if has_object:
                        phrase = ''.join([tok.text_with_ws for tok in token.subtree])
                        phrase = normalize(phrase)
                        if phrase and phrase not in self.position_list and len(phrase.split()) <= 3:
                            self.position_list.append(phrase)
        
        print("The detail list is: ", self.detail_list)
        print("The position list is: ", self.position_list)

        # obtain the attention score
        self.detail_score = self.am.match_clip(self.detail_list, self.showcase, self.base_prompt)
        self.position_score = self.am.match_clip(self.position_list, self.showcase, self.base_prompt)



        detail_score_order = np.argsort(self.detail_score)[::-1]
        position_score_order = np.argsort(self.position_score)[::-1]
        
        self.detail_list = np.array(self.detail_list)
        self.position_list = np.array(self.position_list)

        self.detail_score = np.array(self.detail_score)
        self.position_score = np.array(self.position_score)

        detail_score_order = detail_score_order.tolist()
        position_score_order = position_score_order.tolist()
        
        self.detail_list = self.detail_list[detail_score_order]
        self.position_list = self.position_list[position_score_order]

        self.detail_score = self.detail_score[detail_score_order]
        self.position_score = self.position_score[position_score_order]

        print("The detail_list list is: ", self.detail_list)
        print("The position_list list is: ", self.position_list)

        return

    def obtain_modifier(self):

        modifier_list = load_list(PROJECT_ROOT / "data" / "modifier" / "PS_modifier" / "total.txt")

        modifier_score = self.am.match_clip(modifier_list, self.showcase, self.base_prompt)
        
        self.modifier_list.extend(modifier_list)
        self.modifier_score.extend(modifier_score)

        score_order = np.array(self.modifier_score).argsort()[::-1]

        self.modifier_list = np.array(self.modifier_list)
        self.modifier_score = np.array(self.modifier_score)

        score_order = score_order.tolist()

        self.modifier_list = self.modifier_list[score_order]
        self.modifier_score = self.modifier_score[score_order]

        return

    def generate_candidate(self):

        self.obtain_detail_position()
        self.obtain_modifier()

        return

    def local_candidate(self):
        # merge the detail and position list, sample the top- detail_budget

        detail_list = []
        detail_score = []
        detail_list.extend(self.detail_list)
        detail_list.extend(self.position_list)

        detail_score.extend(self.detail_score)
        detail_score.extend(self.position_score)

        # sort the detail list and reverse the order
        detail_score_order = np.array(detail_score).argsort()[::-1]

        detail_list = np.array(detail_list)
        detail_score = np.array(detail_score)

        detail_score_order = detail_score_order.tolist()

        detail_list = detail_list[detail_score_order]
        detail_score = detail_score[detail_score_order]


        detail_candidate_list = detail_list[:self.detail_budget]
        detail_candidate_score = detail_score[:self.detail_budget]

        # sample the top- modifier_budget
        modifier_candidate_list = self.modifier_list[:self.modifier_budget]
        modifier_candidate_score = self.modifier_score[:self.modifier_budget]

        print(detail_candidate_list.shape[0], detail_candidate_score.shape[0], modifier_candidate_list.shape[0], modifier_candidate_score.shape[0])
        print(detail_candidate_list)
        print(modifier_candidate_list)
        # get the top-k candidates
        return detail_candidate_list, modifier_candidate_list

    def greedy_search(self):

        detail_candidate_list, modifier_candidate_list = self.local_candidate()

        final_list = []
        final_list.extend(detail_candidate_list)
        final_list.extend(modifier_candidate_list)

        spent_budget = 1
        candidate_budget = len(final_list)

        
        self.final_prompt = self.base_prompt
        for num in range(candidate_budget):
            temp_prompt = self.final_prompt + ", " + final_list[num]
            spent_budget += 1

            oracle_feedback = self.query_API.query(temp_prompt)
            print("The spent budget is: ", spent_budget, "The total budget is: ", self.max_budget)

            lpips_similarity = self.evaluator.LPIPS(self.showcase, oracle_feedback)
            ii_similarity = self.evaluator.CLIP_II(self.showcase, oracle_feedback)
            similarity = self.score_fn(lpips_score=lpips_similarity, ii_score=ii_similarity)

            if (similarity-self.best_score) >= self.threshold:
                self.final_prompt = temp_prompt
                if final_list[num] in detail_candidate_list:
                    self.final_dynamic.append(final_list[num])
                elif final_list[num] in modifier_candidate_list:
                    self.final_static.append(final_list[num])
                else:
                    raise ValueError("The final list is not in the detail or modifier list.")
                
                self.best_score = similarity

            else:
                continue

        return self.final_prompt

    def forward(self, showcase):

        self.reset_state()
        self.showcase = showcase
        self.showcase_input = self.transform(self.showcase).unsqueeze(0).to("cuda")

        self.caption_base()
        self.generate_candidate()

        stolen_prompt=self.greedy_search()

        return stolen_prompt

def evaluate(showcase, gt_prompt, final_prompt, online_API, evaluator, metrics):
    generated_image = online_API.query(final_prompt)

    similarity_II = evaluator.CLIP_II([showcase], [generated_image])

    similarity_TT = evaluator.SBERT_TT(gt_prompt, final_prompt)

    similarity_LPIPS = evaluator.LPIPS(showcase, generated_image)


    asr = 1 if similarity_TT > 0.8 else 0

    metrics["II"].update(similarity_II)
    metrics["TT"].update(similarity_TT)
    metrics["LPIPS"].update(similarity_LPIPS)
    metrics["ASR"].update(asr)

    result = {
        "final_prompt": final_prompt,
        "CLIP-II": metric_to_float(similarity_II),
        "LPIPS": metric_to_float(similarity_LPIPS),
        "SBERT": metric_to_float(similarity_TT),
        "ASR": asr,
    }
    return metrics, result


def main():
    args = parse_args()

    dataset = LoadDataset(dataset=args.dataset, oracle=args.oracle, size=args.resolution)
    
    online_API = OnlineModel(model_name=args.oracle)

    evaluator = CLIPEvaluator()

    metrics = {
        "II": MeanMetric(),
        "TT": MeanMetric(),
        "LPIPS": MeanMetric(), 
        "ASR": MeanMetric(),
    }

    start_time = time.time()
    pipe = Prometheus(query_API=online_API, evaluator=evaluator, caption_repeat=args.caption_repeat,
                        max_budget=args.max_budget)
    results = []

    for data in dataset:
        showcase = data["image"]
        gt_prompt = data["prompt"]
        image_name = data["image_name"]

        final_prompt = pipe(showcase)


        metrics, result = evaluate(
            showcase=showcase,
            gt_prompt=gt_prompt,
            final_prompt=final_prompt,
            online_API=online_API,
            evaluator=evaluator,
            metrics=metrics,
        )
        result["showcase_name"] = image_name
        results.append(result)

    final_time = time.time() - start_time

    average_results = {
        "total_time": final_time,
        "CLIP-II": metrics["II"].compute().item(),
        "LPIPS": metrics["LPIPS"].compute().item(),
        "SBERT": metrics["TT"].compute().item(),
        "ASR": metrics["ASR"].compute().item(),
    }

    print("\nPer-prompt results:")
    for result in results:
        print(result)

    print("\nAverage results:")
    print("The total time is: ", final_time)
    print("CLIP-II: ", average_results["CLIP-II"])
    print("LPIPS: ", average_results["LPIPS"])
    print("SBERT: ", average_results["SBERT"])
    print("ASR: ", average_results["ASR"])


if __name__ == "__main__":
    main()
