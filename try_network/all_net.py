import torch
import commons
import torch.nn as nn


class AllNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AllNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, output_dim)
        self.conv1 = torch.nn.Conv1d(1, 80, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(80, 160, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv1d(160, 80, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x,g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) 
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output 

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


def train(x, mel):
    net = WN(
        hidden_channels=1,
        kernel_size=3,
        dilation_rate=2,
        n_layers=6,
        gin_channels=0,
        p_dropout=0.1,
    ).to("cuda")
    net2=AllNet(input_dim=14, output_dim=372).to("cuda")
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.train()
    x, mel = x.cuda(), mel.cuda()
    for epoch in range(10000):
        x=x.unsqueeze(0)  # Add channel dimension
        gen_mel = net(x)
        x=x.squeeze(0)
        gen_mel=net2(gen_mel)
        loss = torch.nn.functional.mse_loss(mel, gen_mel)
        if epoch % 10 == 0:
            print("epoch:", epoch, "loss:", loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    net.eval()
    with torch.no_grad():
        mel = net(x)
        print("mel shape:", mel.shape)
        torch.save(mel, "try.pt")


if __name__ == "__main__":
    mel = torch.load('/home/u-longyy/week2/999.pt').reshape(1, 80, -1)
    x = torch.arange(0, 14).reshape(1, -1).float()
    print("x shape:", x)
    print("mel shape:", mel.shape)
    train(x, mel)
