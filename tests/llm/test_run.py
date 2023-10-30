if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os
import shutil
import tempfile
import unittest

import torch

from swift.llm import DatasetName, InferArguments, ModelType, SftArguments
from swift.llm.run import infer_main, sft_main


class TestRun(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_run_1(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        output_dir = self.tmp_dir
        # output_dir = 'output'
        model_type = ModelType.chatglm2_6b
        sft_args = SftArguments(
            model_type=model_type,
            quantization_bit=4,
            eval_steps=5,
            check_dataset_strategy='warning',
            train_dataset_sample=200,
            predict_with_generate=False,
            dataset=[DatasetName.jd_sentiment_zh],
            output_dir=output_dir,
            gradient_checkpointing=True)
        best_ckpt_dir = sft_main(sft_args)
        print(f'best_ckpt_dir: {best_ckpt_dir}')
        torch.cuda.empty_cache()
        infer_args = InferArguments(
            model_type=model_type,
            quantization_bit=4,
            ckpt_dir=best_ckpt_dir,
            check_dataset_strategy='warning',
            dataset=[DatasetName.jd_sentiment_zh],
            stream=False,
            show_dataset_sample=5,
            merge_lora_and_save=True)
        infer_main(infer_args)

    def test_run_2(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        output_dir = self.tmp_dir
        best_ckpt_dir = sft_main([
            '--model_type',
            ModelType.qwen_7b_chat_int4,
            '--eval_steps',
            '5',
            '--tuner_backend',
            'peft',
            '--train_dataset_sample',
            '200',
            '--predict_with_generate',
            'true',
            '--dataset',
            DatasetName.leetcode_python_en,
            '--output_dir',
            output_dir,
            '--use_flash_attn',
            'false',
            '--gradient_checkpointing',
            'true',
            '--max_new_tokens',
            '100',
        ])
        print(f'best_ckpt_dir: {best_ckpt_dir}')
        torch.cuda.empty_cache()
        infer_main([
            '--model_type',
            ModelType.qwen_7b_chat_int4,
            '--ckpt_dir',
            best_ckpt_dir,
            '--dataset',
            DatasetName.leetcode_python_en,
            '--show_dataset_sample',
            '5',
            '--max_new_tokens',
            '100',
        ])


if __name__ == '__main__':
    unittest.main()