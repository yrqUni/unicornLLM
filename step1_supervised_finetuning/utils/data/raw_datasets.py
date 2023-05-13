from datasets import Dataset
import json

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

# unicorn dataset
class unicorn(PromptRawDataset):

    def __init__(self, output_path, data_input_path, seed, local_rank, eval_split):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "unicorn"
        self.dataset_name_clean = "unicorn"
        self.data_path = data_input_path
        with open(self.data_path, "r") as f:
            data = json.load(f)
        print("data len:", len(data))
        data = Dataset.from_list(data)
        self.raw_datasets = data.train_test_split(test_size=eval_split, shuffle=True)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_conversations(self, sample):
        '''
            for belle data and vicuna data
        '''
        return sample['conversations']

    def get_prompt_and_chosen_alpaca(self, sample):
        if sample['input'] and len(sample['input']) > 0:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Response:
{sample['output']}
"""

    def get_prompt_and_chosen_vicuna(self, sample, sentence_from):
        return "A chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions." + " " \
            + "USR" + ": " + sample["value"] + " " + "ASSISTANT" + ": " if sentence_from == 'human' else sample["value"]
    
    def get_prompt_and_chosen_unicorn(self, sample, sentence_from):
        return 'Human: ' + sample["value"] + '\n\nAssistant: ' if sentence_from == 'human' else sample["value"]
