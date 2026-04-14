import torch.nn as nn
import timm

NUM_CLASSES = 3
CLASS_NAMES = ['healthy', 'early_blight', 'late_blight']


def create_model(model_name, pretrained=True):
    name = model_name.lower()

    if name == 'mobilenetv3':
        m = timm.create_model('mobilenetv3_large_100',
                               pretrained=pretrained,
                               num_classes=0, global_pool='avg')
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.num_features, NUM_CLASSES)
        )

    elif name == 'efficientnet_b0':
        m = timm.create_model('efficientnet_b0',
                               pretrained=pretrained,
                               num_classes=0, global_pool='avg')
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.num_features, NUM_CLASSES)
        )

    elif name == 'vit_tiny':
        m = timm.create_model('vit_tiny_patch16_224',
                               pretrained=pretrained,
                               num_classes=NUM_CLASSES)
        in_f = m.head.in_features
        m.head = nn.Sequential(
            nn.LayerNorm(in_f),
            nn.Dropout(0.3),
            nn.Linear(in_f, NUM_CLASSES)
        )

    elif name == 'mobilevit_s':
        m = timm.create_model('mobilevit_s',
                               pretrained=pretrained,
                               num_classes=0, global_pool='avg')
        m.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.num_features, NUM_CLASSES)
        )

    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    total = sum(p.numel() for p in m.parameters())
    print(f"  {model_name}: {total/1e6:.2f}M parámetros")
    m.model_name = model_name
    return m