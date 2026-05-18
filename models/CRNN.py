import torch
import torch.nn as nn

from models.classification_utils import VectorClassifierHead, to_bcl


# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()  # 使用 GELU 激活
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out

# --- ResNet Encoder ---
class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNetEncoder, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.act = nn.GELU()
        
        self.layer1 = self._make_layer(64, stride=2)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(256, stride=2)

        self.dropout = nn.Dropout(0.1)

    def _make_layer(self, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.dropout(x)

# --- Temporal Attention Mechanism ---
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attn(x)  # (Batch, Seq_Len, 1)
        context = torch.sum(x * weights, dim=1)  # (Batch, Hidden)
        return context

# --- Full Model: ResNet + LSTM + Temporal Attention ---
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.out_channels = int(getattr(configs, "c_out", getattr(configs, "out_in", 1)))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        self.encoder = ResNetEncoder(in_channels=self.in_channels)

        # ResNetEncoder layer4 输出通道数是 256
        cnn_out_channels = 256

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Temporal Attention
        self.attention = TemporalAttention(2)

        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.pred_len * self.out_channels)
        )
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 128))
        cls_dropout = float(getattr(configs, "dropout", 0.2))
        self.cls_head = VectorClassifierHead(2, self.num_classes, hidden_dim=cls_hidden, dropout=cls_dropout)

    def _encode(self, x):
        x = to_bcl(x, self.seq_len, self.in_channels)
        features = self.encoder(x)

        features = features.permute(0, 2, 1)

        out, _ = self.lstm(features)
        context_vector = self.attention(out)
        return context_vector

    def forecast(self, x):
        context_vector = self._encode(x)
        pred = self.fc(context_vector)
        return pred.view(-1, self.pred_len, self.out_channels)

    def classification(self, x):
        context_vector = self._encode(x)
        return self.cls_head(context_vector)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
