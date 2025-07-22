import torch
from create_IPA import convert_sentence_to_codes
def infer(path,model):
    cpt=torch.load(path)
    model.load_state_dict(cpt['model_state_dict'])
    model.eval()
    answer=torch.load("../AItts2mel/melspec/1.pt")
    text=torch.tensor(list(convert_sentence_to_codes("ye51 liaŋ51 thou55 thou55 kei215 xai215 ou55 suŋ51 tɕhy51 niŋ35 məŋ35 khou215 uei51 tɤ0 thaŋ35 kuo215 。")))
    print(text)
    cond=torch.load("../AItts2mel/melspec/1.pt").permute(1,0).unsqueeze(0).cuda()
    text = text.unsqueeze(0).cuda()
    print(answer.shape)
    print(cond.shape)
    mel = model.sample(cond,text,answer.shape[-1],lens=torch.tensor([0]).cuda(),steps=32,no_ref_audio=False)[0]
    mel= mel.permute(0,2,1)
    print("mel shape:", mel.shape)
    torch.save(mel, "try.pt")

from cfm import CFM
from Dit import DiT
if __name__== '__main__':
    model_cls = DiT(
        dim=1024,#1024,
        depth=16,#20,#22,
        heads=16,
        ff_mult=2,
        text_dim=480,#512,
        text_mask_padding=True,
        qk_norm=None,  # null | rms_norm
        conv_layers=4,
        pe_attn_head=None,
        attn_backend="torch",  # torch | flash_attn
        attn_mask_enabled=False,
        checkpoint_activations=False,  # recompute activations and save memory for extra compute)
        text_num_embeds=47,
        mel_dim=80,
    )
    # set model

    model = CFM(
        transformer=model_cls,
    ).cuda()
    infer("F5-tts/new_checkpoint/7_checkpoint.pt",model)