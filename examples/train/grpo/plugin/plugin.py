import re
from typing import List

from swift.plugin import ORM, orms
from swift.utils import get_logger

import spacy
from spacy.tokens import Span
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
import random
import requests

import pdb

logger = get_logger()


class AccuracyORM(ORM):

    def __call__(self, prompts, completions, begin_probs, end_probs, image_paths, answer, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            answer (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, gt in zip(completions, answer):
            reward = 0.0
            try:
                # ground_truth = ",".join([g.split(".")[0].lower() for g in gt.split("\n")])

                # # Extract answer from content if it has think/answer tags
                # content_match = re.search(r'<answer>(.*?)</answer>', content)
                # pred_answer = content_match.group(1).strip() if content_match else ""
                # pred_answer = ",".join([p.split(".")[0].lower() for p in pred_answer.split(",")])
                
                # ground_truth_set = set([g.lstrip().rstrip() for g in ground_truth.split(",")])
                # pred_answer_set = set([p.lstrip().rstrip() for p in pred_answer.split(",")])

                def is_option(s):
                    s = s.strip()
                    # 单个字母
                    if len(s) == 1 and s.isalpha():
                        return True
                    # 以字母或数字开头，后跟. ) : 或空格
                    return bool(re.match(r'^[A-Za-z0-9][\.\):\s]', s))
                if is_option(gt):
                    gt_option = gt.split(".")[0].lower()
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    pred_answer = content_match.group(1).strip() if content_match else ""
                    # pred_option = pred_answer.split(".")[0].lower()
                    pred_option = pred_answer.lstrip().rstrip().split(" ")[0].lower().replace(".", "")

                    # print(pred_option)
                    # print(gt_option)

                    # gt_answer = gt.split(". ")[1].lower().lstrip().rstrip()
                    pred_answer = " ".join(pred_answer.lstrip().rstrip().split(" ")[1:]).lower()

                    # print(pred_answer)
                    # print(gt_answer)

                    # Compare the extracted answers
                    # if pred_answer_set == ground_truth_set:
                    #     reward = 5.0
                        # if len(content) >= 768:
                        #     reward -= 1.0
                    
                    if pred_option == gt_option and len(pred_answer) > 0:
                        reward = 5.0

                    elif pred_option == gt_option:
                        reward = 4.0

                    # elif pred_answer_set.issubset(ground_truth_set) or ground_truth_set.issubset(pred_answer_set):
                    #     reward = 0.5
                    
                    else:
                        reward = 0.0

                        # if len(content) <= 64  or len(content) >= 768:
                        #     reward -= 1.0
                else:
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    pred_answer = content_match.group(1).strip() if content_match else ""
                    if pred_answer.lower() == gt.lower():
                        reward = 5.0
                    else:
                        reward = 0.0



            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards



class LengthORM(ORM):

    def __call__(self, prompts, completions, begin_probs, end_probs, image_paths, answer, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            answer (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, gt in zip(completions, answer):
            # Remove content between <retrieve> and </retrieve> tags
            content_without_retrieve = re.sub(r'<retrieve>.*?</retrieve>', '', content, flags=re.DOTALL)
            
            # if content.lstrip().rstrip().startswith("<answer>"):
            #     reward = 0.0
            # elif 64 < len(content) < 256 :
            #     reward = 0.0
            # else:
            #     reward = -1

            if 512 <= len(content_without_retrieve) <= 3072:
                reward = 0
            else:
                reward = -1
            rewards.append(reward)
        return rewards

class umls_extractor:
    """
    Extracts medical entities from text using scispacy and UMLS
    """
    def __init__(self):
        """Initialize the NLP pipeline with medical entity linking capabilities"""
        nlp = spacy.load("en_core_sci_lg")
        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.nlp = nlp
        
    def extract(self, text):
        """
        Extract medical entities from text
        
        Args:
            text: Input text to process
            
        Returns:
            Set of extracted entities
        """
        doc = self.nlp(text)
        ent_set = doc.ents
        return ent_set


class RetrieveSemanticORM(ORM):
    def __init__(self):
        self.word_extract = umls_extractor()    

    def count_overlap_medical_words_portion(self, text, answer_ents):
        word_count = 0
        total_word_count = 0

        for ent in answer_ents:
            for e in ent.split(" "):
                if e in text:
                    word_count += 1
                total_word_count += 1
        if total_word_count == 0:
            return 0.0
        return word_count * 1.0 / total_word_count

    def __call__(self, prompts, completions, begin_probs, end_probs, image_paths, answer, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            answer (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        
        rewards = []
        for prompt, content, gt in zip(prompts, completions, answer):
            reward, penalty = 0.0, 0.0
        
            gt = ", ".join([g.split(".")[0].lower() for g in gt.split("\n")])
            answer_ents = list(set([str(e) for e in self.word_extract.extract(gt)]))
            
            if "<query>" in content and "</query>" in content:
                # queries = re.findall(r'<query>(.*?)</query>', content, re.DOTALL)
                # query = queries[0].lower() if queries else None
                queries = re.findall(r'<query>(.*?)</query>', content, re.DOTALL)
                # 合并所有query
                query = ' '.join([q.lower() for q in queries]) if queries else None
                if query:
                    query_ents = list(set([str(e) for e in self.word_extract.extract(query)]))

                    if len(query_ents) > 5:
                        penalty = 1
                    else:
                        penalty = 0

                    query_portion = self.count_overlap_medical_words_portion(query, answer_ents)
                    query_ent_portion = sum(len(e.split(" ")) for e in query_ents) * 1.0 / len(query.split(" "))
                else:
                    query_portion = 0
                    query_ent_portion = 0

                retrieved = re.findall(r'<retrieve>(.*?)</retrieve>', prompt, re.DOTALL)
                retrieved = retrieved[-1].lower() if retrieved else None
                if retrieved:
                    retrieved_portion = self.count_overlap_medical_words_portion(retrieved, answer_ents)
                else:
                    retrieved_portion = 0

                reward = (retrieved_portion + query_portion + query_ent_portion) - penalty
                # reward = retrieved_portion * 5.0

            rewards.append(reward)
        return rewards

    
class RetrieveImageSimORM(ORM):
    def __init__(self):
        self.word_extract = umls_extractor()    

    def count_overlap_medical_words_portion(self, text, answer_ents):
        word_count = 0
        total_word_count = 0

        for ent in answer_ents:
            for e in ent.split(" "):
                if e in text:
                    word_count += 1
                total_word_count += 1
        if total_word_count == 0:
            return 0.0
        return word_count * 1.0 / total_word_count

    def __call__(self, prompts, completions, begin_probs, end_probs, image_paths, answer, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            answer (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        
        rewards = []
        query_list = []
        for prompt, content, gt in zip(prompts, completions, answer):
            reward = 0.0

            # if there is a query, use the query content to compute the similarity with input image
            if "<query>" in content and "</query>" in content:
                queries = re.findall(r'<query>(.*?)</query>', content, re.DOTALL)
                query = random.choice(queries).lower() if queries else None
            # if there is no query, use the first sentence to compute the similarity with input image
            else:
                query = content.split(".")[0].replace("<think>", "").replace("</think>", "").replace("<query>", "").replace("</query>", "").replace("<retrieve>", "").replace("</retrieve>", "").replace("<answer>", "").replace("</answer>", "").lstrip().rstrip().lower()

            query_list.append(query)

        clip_url = "http://127.0.0.1:5000/biomedclip"
        response = requests.post(clip_url, json={"test_imgs": image_paths, "labels": query_list})
        rewards = [s * 0.1 for s in response.json()['similarity_scores']]
            
        return rewards


class MultiStepFormatORM(ORM):
    def check_empty_blocks(self, content):
        # Returns 1 if there is any empty or whitespace-only block between tags, else 0
        tag_patterns = [
            r'<think>(.*?)</think>',
            r'<query>(.*?)</query>',
            r'<retrieve>(.*?)</retrieve>',
            r'<answer>(.*?)</answer>',
        ]
        for pattern in tag_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if match.strip() == '':
                    return 1
        return 0

    def check_consecutive_newlines(self, content):
        # Returns 1 if there are three or more consecutive newlines, else 0
        return 1 if re.search(r'[\n\r]{3,}', content) else 0

    def has_malformed_tokens(self, content):
        # Disallow any substring like =word>
        if re.search(r'=[a-zA-Z]+>', content):
            return True
        # Disallow any tag-like structure not in the allowed set
        allowed_tags = {
            "<think>", "</think>", "<query>", "</query>",
            "<retrieve>", "</retrieve>", "<answer>", "</answer>"
        }
        for tag in re.findall(r'<[^>]+>', content):
            if tag not in allowed_tags:
                return True
        return False

    def retrieve_without_query(self, content):
        # Returns True if a <retrieve> block appears without a preceding <query> block
        # We'll scan for <query> and <retrieve> tags in order
        pattern = r'(<query>.*?</query>|<retrieve>.*?</retrieve>)'
        blocks = re.findall(pattern, content, re.DOTALL)
        last_was_query = False
        for block in blocks:
            if block.startswith('<query>'):
                last_was_query = True
            elif block.startswith('<retrieve>'):
                if not last_was_query:
                    return True
                last_was_query = False
        return False

    def __call__(self, prompts, completions, begin_probs, end_probs, image_paths, answer, **kwargs) -> List[float]:
        """
        Reward function for completions with the format:
        (<think>...</think>(<query>...</query><retrieve>...</retrieve>)?)*<answer>...</answer>
        The <answer>...</answer> must only appear at the end.
        Now also allows only <think> blocks (without <query> and <retrieve>), but in that case <think> must appear exactly once.
        Penalizes empty content between tags and three or more consecutive newlines (penalty is at most 1 for each type).
        If any malformed tag is present, the format is considered invalid and reward is 0.
        If a <retrieve> block appears without a preceding <query> block, the reward is 0.
        """
        rewards = []
        repeated_block = r'(?:<think>[^<>]*?</think>(?:\s*<query>[^<>]*?</query>\s*<retrieve>[^<>]*?</retrieve>)?\s*)*'
        answer_block = r'<answer>[^<>]*?</answer>\s*$'
        full_pattern = f'^{repeated_block}{answer_block}'
        answer_tag_pattern = r'<answer>.*?</answer>'
        think_tag_pattern = r'<think>.*?</think>'
        query_tag_pattern = r'<query>.*?</query>'
        retrieve_tag_pattern = r'<retrieve>.*?</retrieve>'

        for content, gt in zip(completions, answer):
            # Check for malformed tokens first
            if self.has_malformed_tokens(content):
                rewards.append(0.0)
                continue
            # Check for <retrieve> without preceding <query>
            if self.retrieve_without_query(content):
                rewards.append(0.0)
                continue
            # try:
            #     ground_truth = ",".join([g.split(".")[0].lower() for g in gt.split("\n")])
            #     content_match = re.search(r'<answer>(.*?)</answer>', content)
            #     pred_answer = content_match.group(1).strip() if content_match else ""
            #     pred_answer = ",".join([p.split(".")[0].lower() for p in pred_answer.split(",")])
            #     ground_truth_set = set([g.lstrip().rstrip().replace("\n", "") for g in ground_truth.split(",")])
            #     pred_answer_set = set([p.lstrip().rstrip().replace("\n", "") for p in pred_answer.split(",")])
            #     is_correct = pred_answer_set == ground_truth_set
            # except Exception:
            #     is_correct = False

            try:
                def is_option(s):
                    s = s.strip()
                    # 单个字母
                    if len(s) == 1 and s.isalpha():
                        return True
                    return bool(re.match(r'^[A-Za-z0-9][\.\):\s]', s))
                if is_option(gt):
                    gt_option = gt.split(".")[0].lower()
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    pred_answer = content_match.group(1).strip() if content_match else ""
                    pred_option = pred_answer.lstrip().rstrip().split(" ")[0].lower().replace(".", "")
                    pred_answer = " ".join(pred_answer.lstrip().rstrip().split(" ")[1:]).lower()
                    if pred_option == gt_option and len(pred_answer) > 0:
                        is_correct = True    
                    else:
                        is_correct = False
                else:
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    pred_answer = content_match.group(1).strip() if content_match else ""
                    if pred_answer.lower() == gt.lower():
                        is_correct = True
                    else:
                        is_correct = False
            except Exception:
                is_correct = False
        

            num_think = len(re.findall(think_tag_pattern, content, re.DOTALL | re.MULTILINE))
            num_query = len(re.findall(query_tag_pattern, content, re.DOTALL | re.MULTILINE))
            num_retrieve = len(re.findall(retrieve_tag_pattern, content, re.DOTALL | re.MULTILINE))
            answer_tags = list(re.finditer(answer_tag_pattern, content, re.DOTALL | re.MULTILINE))
            if num_query == 0 and num_retrieve == 0:
                pattern = r'^<think>[^<>]*?</think>\s*<answer>[^<>]*?</answer>\s*$'
                match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
                if match and num_think == 1 and len(answer_tags) == 1 and answer_tags[0].end() == len(content.strip()):
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                match = re.match(full_pattern, content, re.DOTALL | re.MULTILINE)
                if match and len(answer_tags) == 1 and answer_tags[0].end() == len(content.strip()):
                    if is_correct:
                        reward = 3.0
                    else:
                        reward = 1.0
                else:
                    reward = 0.0
            # add penalty for empty content between tags
            penalty_empty = self.check_empty_blocks(content)
            # add penalty for three or more consecutive newlines
            penalty_newlines = self.check_consecutive_newlines(content)
            reward -= (penalty_empty + penalty_newlines)
            rewards.append(reward)
        return rewards


class RetrieveLogitORM(ORM):

    def __call__(self, prompts, completions, begin_probs, end_probs, image_paths, answer, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            answer (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, gt, begin_prob, end_prob in zip(completions, answer, begin_probs, end_probs):
            gt_option = gt.split(".")[0].lower()

            content_match = re.search(r'<answer>(.*?)</answer>', content)
            pred_answer = content_match.group(1).strip() if content_match else ""
            pred_option = pred_answer.lstrip().rstrip().split(" ")[0].lower().replace(".", "")

            begin_rank = 0
            end_rank = 10
            begin_prob_score = 0
            end_prob_score = 0

            for i, prob in enumerate(begin_prob):
                if prob['token'] == gt_option:
                    begin_prob_score = prob['logprob']
                    begin_rank = i
                    break

            for j,prob in enumerate(end_prob):
                if prob['token'] == gt_option:
                    end_prob_score = prob['logprob']
                    end_rank = j
                    break
            
            if pred_option == gt_option:
                reward = (end_prob_score - begin_prob_score) * 0.1
            else:
                reward = (begin_rank - end_rank) * 0.05

            rewards.append(reward)
        return rewards

orms['external_acc'] = AccuracyORM
orms['external_length'] = LengthORM
orms['external_retrieve_semantic'] = RetrieveSemanticORM
orms['external_retrieve_format'] = MultiStepFormatORM
orms['external_retrieve_logit'] = RetrieveLogitORM
orms['external_retrieve_imagesim'] = RetrieveImageSimORM