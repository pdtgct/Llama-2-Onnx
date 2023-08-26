# This program will run the ONNX version of the LlamaV2 model.
import torch
import onnxruntime
import numpy as np
from sentencepiece import SentencePieceProcessor
from typing import List
import os
import argparse


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

class LLMSession:

    def __init__(self, onnx_file: str, embedding_file: str, tokenizer_path: str):
        # Create the ONNX session
        options = onnxruntime.SessionOptions()
        self.llm_session = onnxruntime.InferenceSession(
            onnx_file,
            sess_options=options,
            providers=[
                # DML is for Windows
                # "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        # get the data type used by the model
        data_type_str = self.llm_session.get_inputs()[0].type
        if data_type_str == "tensor(float16)":
            self.data_type = np.float16
        elif data_type_str == "tensor(float32)" or data_type_str == "tensor(float)":
            self.data_type = np.float32
        else:
            raise Exception(f"Unknown data type {data_type_str}")

        # Get the relevant shapes so we can create the inputs
        for inputs_meta in self.llm_session._inputs_meta:
            if inputs_meta.name == "x":
                x_shape = inputs_meta.shape
            elif inputs_meta.name == "attn_mask":
                attn_mask_shape = inputs_meta.shape
            elif inputs_meta.name == "k_cache":
                k_cache_shape = inputs_meta.shape

        hidden_size = x_shape[2]
        self.max_seq_len = attn_mask_shape[1]
        n_layers = k_cache_shape[1]
        n_heads = k_cache_shape[3]

        # Initialize the tokenizer and produce the initial tokens.
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        # Create the attention mask.
        self.attn_mask = -10000.0 * torch.triu(
            torch.ones(attn_mask_shape), diagonal=1
        ).cpu().detach().numpy().astype(self.data_type)

        # create the embedding layer.
        self.embedding_layer = torch.nn.Embedding(self.tokenizer.n_words, hidden_size)
        self.embedding_layer.load_state_dict(torch.load(embedding_file))
        self.embedding_layer.eval()

        # Create the K and V caches.
        head_dim = int(hidden_size / n_heads)
        self.k_cache = np.zeros([1, n_layers, self.max_seq_len, n_heads, head_dim], dtype=self.data_type)
        self.v_cache = np.zeros([1, n_layers, self.max_seq_len, n_heads, head_dim], dtype=self.data_type)

    def run_onnx(
        self,
        prompt: str,
        max_gen_len: int = 256,
    ) -> str:

        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        # Create the embeddings of the initial prompt.
        x = self.embedding_layer(torch.tensor(tokens)).detach().cpu().numpy()
        x = np.expand_dims(x, axis=0).astype(self.data_type)

        # Iteratively generate tokens.
        pos = np.array(0)
        output_tokens = []
        for idx in range(max_gen_len):
            results = self.llm_session.run(
                None,
                {
                    "x": x,
                    "attn_mask": self.attn_mask,
                    "k_cache": self.k_cache[:, :, :pos],
                    "v_cache": self.v_cache[:, :, :pos],
                    "pos": pos.astype(np.int64),
                },
            )
            logits, k_out, v_out = results[:3]

            # Decide the next token using your preferred sampling strategy.
            next_token = np.argmax(logits, axis=-1).astype(np.int64)
            output_tokens.extend(next_token)

            # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
            if next_token == self.tokenizer.eos_id:
                break

            # Update the cache
            seq_len = x.shape[1]
            self.k_cache[:, :, pos : pos + seq_len] = k_out
            self.v_cache[:, :, pos : pos + seq_len] = v_out

            # Update pos and x ready for the next round.
            pos = np.array(int(pos) + seq_len, dtype=np.int64)
            x = self.embedding_layer(torch.tensor(next_token)).unsqueeze(0)
            x = x.cpu().detach().numpy().astype(self.data_type)

        output_tokens = torch.tensor(output_tokens).tolist()
        output_str = self.tokenizer.decode(output_tokens)

        return output_str, output_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--onnx_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
    )
    parser.add_argument("--max_gen_len", type=int, default=256)
    args = parser.parse_args()
    sess = LLMSession(
        args.onnx_file, 
        args.embedding_file,
        args.tokenizer_path,
    )
    response = sess.run_onnx(
        args.prompt,
        args.max_gen_len,
    )

    print(response)
