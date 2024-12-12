from collections import OrderedDict
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, c, h, w = x.shape

        # Create grid of shape (h, w)
        y_embed = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w)
        # print(y_embed.size())
        x_embed = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1)
        # print(x_embed.size())
        # Normalize by the dimensions of the feature map
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Compute positional encodings for x and y
        dim_t = torch.arange(self.hidden_dim // 2, device=x.device).float()
        dim_t = 10000 ** (2 * (dim_t // 2) / self.hidden_dim)
        # print(dim_t.size())
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        # print(pos_x.size(), pos_y.size())
        
        # pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        # pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        # print(pos_x.size(), pos_y.size())
        

        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1)
        return pos.unsqueeze(0).repeat(b, 1, 1, 1)


class Transformer(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, 
                 num_decoder_layers=6):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, ),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads),
            num_layers=num_decoder_layers
        )

    def forward(self, src, pos, query_embed):
        b, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) + pos.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src)
        # print(memory.shape)
        query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(tgt, memory)
        # print(hs.transpose(0, 1).shape)
        return hs.transpose(0, 1)



class DETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256, nheads=8, device='cuda'):
        super().__init__()
        self.device = device
        # Move backbone to specified device
        self.backbone = self._create_backbone('resnet50', pretrained=True, device=self.device)
        
        # Move all other components to the same device
        self.conv = nn.Conv2d(2048, hidden_dim, 1, 1).to(self.device)
        self.positional_encoding = PositionalEncoding(hidden_dim).to(self.device)
        self.transformer = Transformer(hidden_dim, nheads).to(self.device)
        
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim).to(self.device)
        
        self.dropout = nn.Dropout(0.2)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1).to(self.device)
        self.bbox_embed = nn.Linear(hidden_dim, 4).to(self.device)

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Only check parametric layers
        parametric_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        for name, module in self.backbone.named_children():
            x = module(x)
        
        features = x  # Use the processed features
        # print(features.shape)
        # sys.exit()
        features = self.conv(features)
        pos = self.positional_encoding(features)
        
        hs = self.transformer(features, pos, self.query_embed.weight)
        hs = self.dropout(hs)
        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()
        return outputs_class, outputs_bbox

    def _create_backbone(self, backbone_name, pretrained, device):
        if backbone_name == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            
            # Move to device
            backbone = backbone.to(device)
            
            backbone = nn.Sequential(OrderedDict([
                ('conv1', backbone.conv1),
                ('bn1', backbone.bn1),
                ('relu', backbone.relu),
                ('maxpool', backbone.maxpool),
                ('layer1', backbone.layer1),
                ('layer2', backbone.layer2),
                ('layer3', backbone.layer3),
                ('layer4', backbone.layer4)
            ]))
            
            return backbone
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
