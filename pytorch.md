### squeeze
a.squeeze(dim=None)移除尺寸为1的指定维度,如果不指定维度则默认移除所有尺寸为1的维度

### unsqueeze
a.unsqueeze(dim),增加一个尺寸为1的指定维度

### detach
a.detach()把a从计算图中分离出来，相当于创建一个a的副本

### 生成掩码
``` python
def sequence_mask(length, max_length=None):
  ''' length: [batch_size,seq_len]'''
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)#返回一个前面是True后面是False的张量
```