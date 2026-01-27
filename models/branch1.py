import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch 
#from torchsummary import summary

class DipResNet2Layers(nn.Module):
    def __init__(
        self,
        num_input_channels= 10,
        num_output_channels= 11,
        num_channels= 64,
        act_fun="LeakyReLU",
        norm_layer=nn.BatchNorm2d,
        pad="reflection",
    ):
        super(DipResNet2Layers, self).__init__()

        # --- Layer 1: Two Residual Blocks ---
        self.layer1 = nn.Sequential(
            self.conv(num_input_channels, num_channels, 3, stride=1, pad=pad),
            self.act(act_fun),
            *self.get_block(num_channels, num_channels, norm_layer, act_fun),  # Block 1
            #*self.get_block(num_channels, num_channels, norm_layer, act_fun),  # Block 2
        )

        # --- Layer 2: Two Residual Blocks ---
        self.layer2 = nn.Sequential(
            *self.get_block(num_channels, num_channels, norm_layer, act_fun),  # Block 3
            #*self.get_block(num_channels, num_channels, norm_layer, act_fun),  # Block 4
        )

        self.layer3 = nn.Sequential(
            *self.get_block(num_channels, num_channels, norm_layer, act_fun),  # Block 3
            #*self.get_block(num_channels, num_channels, norm_layer, act_fun),  # Block 4
        )

        # --- Final Output Layer ---
        self.final = self.conv(num_channels, num_output_channels, 1, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x, self.final(x)

    # --- Helper Functions (Same as Before) ---
    def act(self, act_fun="LeakyReLU"):
        if isinstance(act_fun, str):
            if act_fun == "LeakyReLU":
                return nn.LeakyReLU(0.2, inplace=True)
            elif act_fun == "ReLU":
                return nn.ReLU(inplace=True)
            elif act_fun == "ELU":
                return nn.ELU(inplace=True)
            elif act_fun == "none":
                return nn.Sequential()
            else:
                raise ValueError(f"Unknown activation: {act_fun}")
        return act_fun

    def conv(self, in_f, out_f, kernel_size, stride=1, bias=True, pad="zero"):
        to_pad = (kernel_size - 1) // 2
        if pad == "reflection":
            padder = nn.ReflectionPad2d(to_pad)
        elif pad == "zero":
            padder = nn.ZeroPad2d(to_pad)
        else:
            raise ValueError(f"Unknown padding: {pad}")
        
        conv = nn.Conv2d(in_f, out_f, kernel_size, stride=stride, padding=0, bias=bias)
        return nn.Sequential(padder, conv)

    def get_block(self, num_channels_in, num_channels, norm_layer, act_fun):
        return [
            nn.Conv2d(num_channels_in, num_channels, 3, stride=1, padding=1, bias=False),
            norm_layer(num_channels, affine=True),
            self.act(act_fun),
            nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1, bias=False),
            norm_layer(num_channels, affine=True),
        ]

if __name__ == "__main__":
    # --- Test Parameters ---
    num_input_channels = 10  # RGB input
    num_output_channels = 11  # RGB output
    num_channels = 64        # Intermediate channels
    input_size = (224, 224)  # Spatial dimensions

    # --- Initialize Model ---
    model = DipResNet2Layers(
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        num_channels=num_channels,
        act_fun="LeakyReLU",
        pad="reflection"
    )

    # # --- Print Model Summary (requires torchsummary) ---
    # try:
    #     print("\nModel Summary:")
    #     summary(model, input_size=(num_input_channels, *input_size), device="cpu")
    # except ImportError:
    #     print("\nInstall torchsummary for detailed layer info: `pip install torchsummary`")
    #     print("Model architecture:")
    #     print(model)

    # --- Test Forward Pass ---
    dummy_input = torch.randn(1, num_input_channels, *input_size)  # Batch of 1
    output = model(dummy_input)

    print("\nInput Shape:", dummy_input.shape)
    print("Output Shape:", output.shape)  # Should match input spatial size
    assert output.shape[2:] == input_size, "Spatial dimensions changed!"