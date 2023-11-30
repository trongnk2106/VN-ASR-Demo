from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, load_metric, Dataset, ClassLabel#, Audio
import soundfile as sf
import pandas as pd
# from transformers import cached_path, hf_bucket_url
from transformers.utils.hub import cached_file
import os, zipfile
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import IPython
import torchaudio
import torch
import json

import random

if not os.path.exists("./store_weight"):
    os.makedirs("./store_weight")
    
cache_dir = './store_weight/'
# if not have pretrain please uncomment 2 rows below
# processor = Wav2Vec2Processor.from_pretrained("foxxy-hm/wav2vec2-base-finetune-vi-v6", cache_dir=cache_dir)
# model = Wav2Vec2ForCTC.from_pretrained("foxxy-hm/wav2vec2-base-finetune-vi-v6", cache_dir=cache_dir)

processor = Wav2Vec2Processor.from_pretrained("./store_weight/models--foxxy-hm--wav2vec2-base-finetune-vi-v6/snapshots/e4dc13d3d6cc3fb2f84f92dc483d48fd772ae695", cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained("./store_weight/models--foxxy-hm--wav2vec2-base-finetune-vi-v6/snapshots/e4dc13d3d6cc3fb2f84f92dc483d48fd772ae695", cache_dir=cache_dir)

class S2T_Model:

    @staticmethod
    def map_to_array(batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        batch["sampling_rate"] = 16000

        return batch

    @staticmethod
    def predict(path_file):
        ds = S2T_Model.map_to_array({"file": path_file})
        input_values = processor(ds["speech"], return_tensors="pt", padding="longest").input_values  # Batch size 1
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription

    
    
