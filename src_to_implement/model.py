import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)

        self.skip_connection = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, input_tensor):
        skip_tensor = self.skip_connection(input_tensor)
        conv1_output = self.conv1(input_tensor)
        batch_norm1_output = self.batch_norm1(conv1_output)
        relu_output = self.relu(batch_norm1_output)
        conv2_output = self.conv2(relu_output)
        batch_norm2_output = self.batch_norm2(conv2_output)
        output_tensor = self.relu(batch_norm2_output + skip_tensor)
        return output_tensor


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.initial_batch_norm = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_layer1 = self._make_residual_layer(64, 64, num_blocks=2, stride=1)
        self.residual_layer2 = self._make_residual_layer(64, 128, num_blocks=2, stride=2)
        self.residual_layer3 = self._make_residual_layer(128, 256, num_blocks=2, stride=2)
        self.residual_layer4 = self._make_residual_layer(256, 512, num_blocks=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        self.output_activation = nn.Sigmoid()

    def _make_residual_layer(self, input_channels, output_channels, num_blocks, stride):
        residual_layers = []
        residual_layers.append(ResidualBlock(input_channels, output_channels, stride))
        for _ in range(1, num_blocks):
            residual_layers.append(ResidualBlock(output_channels, output_channels))
        return nn.Sequential(*residual_layers)

    def forward(self, input_tensor):
        conv_output = self.initial_conv(input_tensor)
        batch_norm_output = self.initial_batch_norm(conv_output)
        relu_output = self.initial_relu(batch_norm_output)
        maxpool_output = self.initial_maxpool(relu_output)

        residual_output1 = self.residual_layer1(maxpool_output)
        residual_output2 = self.residual_layer2(residual_output1)
        residual_output3 = self.residual_layer3(residual_output2)
        residual_output4 = self.residual_layer4(residual_output3)

        avg_pool_output = self.global_avg_pool(residual_output4)
        flatten_output = self.flatten(avg_pool_output)
        fc_output = self.fc(flatten_output)
        final_output = self.output_activation(fc_output)
        return final_output
