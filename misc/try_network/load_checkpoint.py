from model import SynthesizerTrn
from create_IPA import convert_sentence_to_codes
import torch
def load_check(path):
    checkpoint = torch.load(path)
    model=SynthesizerTrn(
            n_vocab=47,
            spec_channels=80,
            inter_channels=192,
            hidden_channels=192,#192,
            filter_channels=792,
            n_heads=4,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            use_sdp=False,
        ).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    
if __name__ == '__main__':
    model=load_check('checkpoint/100_checkpoint.pt')
    model.eval()
    with torch.no_grad():
        x_hat=torch.tensor(list(convert_sentence_to_codes("fəŋ55 tɕiaŋ55 tɕi51 tɕiŋ51 xua51 uei51 u35 ʂu51 ɕi51 suei51 tɤ0 tsuan55 ʂʅ35 。")))
        x = torch.zeros(135,dtype=torch.long)
        x[:len(x_hat)]=x_hat
        x_lengths = torch.tensor([len(x_hat)]).cuda()
        x = x.unsqueeze(0).cuda()
        mel = model.infer(x, x_lengths)[0]
        print("mel shape:", mel.shape)
        torch.save(mel, "try.pt")