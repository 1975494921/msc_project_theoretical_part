import glob
import pandas as pd
from typing import Union, List, Literal
import numpy as np
import json

from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput

class MathCodeDataset(BaseDataset):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets/Math_Code/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        jsonl_paths = glob.glob(data_path + "*.jsonl")
        jsonl_paths = sorted(jsonl_paths)
        print("Number of topics: ", len(jsonl_paths))

        total_data = []
        for path in jsonl_paths:
            with open(path, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    total_data.append(record)

        total_df = pd.DataFrame(total_data)

        # Ensure the DataFrame has the expected columns
        expected_columns = ['question', 'A', 'B', 'C', 'D', 'correct_answer']
        if not all(column in total_df.columns for column in expected_columns):
            raise ValueError(f"Data is missing one or more of the required columns: {expected_columns}")

        # Pseudorandom shuffle
        total_df = total_df.sample(frac=1, random_state=rng).reset_index(drop=True)

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_swarm_input(record: pd.DataFrame) -> SwarmInput:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
