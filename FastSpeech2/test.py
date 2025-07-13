import torch
import yaml
from model import FastSpeech2

if __name__ == "__main__":
    # Example usage
    preprocess_config = yaml.safe_load(open('config/preprocess.yaml', 'r'))
    model_config = yaml.safe_load(open('config/model.yaml', 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FastSpeech2(preprocess_config, model_config).to(device)

    # speakers = torch.tensor([0, 0]).to(device)
    texts = torch.randint(0, 100, (2, 50)).to(device)  # Example text input
    src_lens = torch.tensor([50, 45]).to(device)  # Example source lengths
    max_src_len = 50
    output = model(texts, src_lens, max_src_len)
    print(output[0].shape)  # Output mel-spectrograms