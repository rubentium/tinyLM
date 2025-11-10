import torch
import torch.nn as nn
import torchao.quantization as quant
from dataclasses import dataclass, asdict

from src.transformer import Model
from src.dataloader import get_data_loaders

@dataclass
class HyperParams:
    batch_size: int = 4
    embed_size: int = 32
    ff_hidden_size: int = 64
    num_layers: int = 2
    seq_len: int = 64

    global_steps: int = 1000
    minibatch: int = 4
    eval_freq: int = 100
    lr: float = 0.001
    lr_min: float = 0.0001


class Generator:
    def __init__(self):
        _, _, meta = get_data_loaders(batch_size=32, seq_len=128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        ckpt_path = "tinyLM/checkpoints/tinyLM_1762703450.ckpt"
        checkpoint = torch.load(ckpt_path)
        hyperparams = HyperParams(**checkpoint['hyperparams'])
        self.model = Model(meta["vocab_size"], 
                           embed_size=hyperparams.embed_size, 
                           ff_hidden_size=hyperparams.ff_hidden_size, 
                           num_layers=hyperparams.num_layers, 
                           seq_len=hyperparams.seq_len,
                            num_heads=hyperparams.num_heads
                           ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = meta["tokenizer"]
        print(hyperparams)

    def dynamic_quantization(self):
        self.model.eval()
        quant_config = quant.Int8DynamicActivationInt4WeightConfig()
        quant.quantize_(self.model, quant_config)
        print("Model quantized")
    
    def static_quantization(self):
        self.model.eval()


    def generate_text(self, prompt, max_new_tokens=5, temperature=1.0):
        input_tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        generated, time = self.model.generate(input_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text, time

if __name__ == "__main__":
    gen = Generator()
    gen.dynamic_quantization()
    prompt = "Once upon a time"
    generated_text, time = gen.generate_text(prompt, max_new_tokens=15, temperature=0.8)
    print(generated_text)
    print(f"Generation time: {1000*time:.1f} miliseconds")