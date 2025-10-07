import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración optimizada ---
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"¿GPU disponible?: {torch.cuda.is_available()}")
print(f"Nombre de la GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Hiperparámetros ajustados
BATCH_SIZE = 4
EPOCHS = 100
LR = 6e-5
IMAGE_SIZE = (202, 202)
NUM_CLASSES = 4  # 0 fondo + 3 clases

# --------------------------
# --- Implementación SegFormer
# --------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=7, stride=4, in_chans=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = x.transpose(1, 2).view(-1, x.size(-1), H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class SegFormerEncoder(nn.Module):
    def __init__(self, img_size=128, in_chans=1, embed_dims=[32,64,160,256], num_heads=[1,2,5,8], mlp_ratios=[4,4,4,4], qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths=[2,2,2,2], sr_ratios=[8,4,2,1]):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size//4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size//8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([TransformerBlock(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0]); cur += depths[0]
        self.block2 = nn.ModuleList([TransformerBlock(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1]); cur += depths[1]
        self.block3 = nn.ModuleList([TransformerBlock(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2]); cur += depths[2]
        self.block4 = nn.ModuleList([TransformerBlock(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
    def forward(self, x):
        B = x.shape[0]; outs = []
        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        x = self.norm1(x); x = x.reshape(B, H, W, -1).permute(0,3,1,2).contiguous(); outs.append(x)
        x, H, W = self.patch_embed2(x)
        for blk in self.block2: x = blk(x, H, W)
        x = self.norm2(x); x = x.reshape(B, H, W, -1).permute(0,3,1,2).contiguous(); outs.append(x)
        x, H, W = self.patch_embed3(x)
        for blk in self.block3: x = blk(x, H, W)
        x = self.norm3(x); x = x.reshape(B, H, W, -1).permute(0,3,1,2).contiguous(); outs.append(x)
        x, H, W = self.patch_embed4(x)
        for blk in self.block4: x = blk(x, H, W)
        x = self.norm4(x); x = x.reshape(B, H, W, -1).permute(0,3,1,2).contiguous(); outs.append(x)
        return outs

class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels, decoder_embed_dim, num_classes):
        super().__init__()
        self.proj_layers = nn.ModuleList()
        self.adapt_layers = nn.ModuleList()
        for channels in in_channels:
            self.proj_layers.append(nn.Conv2d(channels, decoder_embed_dim, kernel_size=1))
            self.adapt_layers.append(nn.Sequential(nn.Conv2d(decoder_embed_dim, decoder_embed_dim, kernel_size=3, padding=1), nn.BatchNorm2d(decoder_embed_dim), nn.ReLU(inplace=True)))
        self.fusion = nn.Sequential(nn.Conv2d(decoder_embed_dim * len(in_channels), decoder_embed_dim, 1), nn.BatchNorm2d(decoder_embed_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Conv2d(decoder_embed_dim, num_classes, 1)
        self.up_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    def forward(self, features):
        proj_features = [self.proj_layers[i](feat) for i, feat in enumerate(features)]
        target_size = proj_features[0].shape[2:]
        resized_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) if f.shape[2:] != target_size else f for f in proj_features]
        adapted_features = [self.adapt_layers[i](feat) for i, feat in enumerate(resized_features)]
        fused = torch.cat(adapted_features, dim=1)
        fused = self.fusion(fused)
        output = self.classifier(fused)
        output = self.up_final(output)
        return output

class SegFormer(nn.Module):
    def __init__(self, in_chans=1, num_classes=4):
        super().__init__()
        self.encoder = SegFormerEncoder(img_size=128, in_chans=in_chans, embed_dims=[32,64,160,256], num_heads=[1,2,5,8], mlp_ratios=[4,4,4,4], depths=[2,2,2,2], sr_ratios=[8,4,2,1])
        self.decoder = SegFormerDecoder(in_channels=[32,64,160,256], decoder_embed_dim=128, num_classes=num_classes)
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out

# --------------------------
# --- Dataset multiclase (3 máscaras por imagen, mismo nombre)
# --------------------------
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dirs, image_transform=None, mask_transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        if isinstance(masks_dirs, str):
            masks_dirs = [masks_dirs]
        self.masks_dirs = masks_dirs
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # verificar que en cada carpeta exista la máscara con mismo nombre
        for mdir in masks_dirs:
            for img_path in self.image_paths:
                name = os.path.basename(img_path)
                if not os.path.exists(os.path.join(mdir, name)):
                    raise FileNotFoundError(f"No existe máscara {name} en {mdir}")

        # flag para guardar solo una vez
        self.saved_debug_image = False  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import matplotlib.pyplot as plt
        import numpy as np

        img_path = self.image_paths[idx]
        name = os.path.basename(img_path)

        # imagen (grayscale)
        image = Image.open(img_path).convert("L")
        if self.image_transform:
            image_tensor = self.image_transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        # combinar 3 máscaras en una sola máscara multi-clase
        combined_mask = None
        for class_idx, mdir in enumerate(self.masks_dirs, start=1):
            mask_path = os.path.join(mdir, name)
            mask_img = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask_img = self.mask_transform(mask_img)  # tensor [1,H,W]
            mask_tensor = mask_img.squeeze(0)  # [H,W]
            if combined_mask is None:
                combined_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
            combined_mask[mask_tensor > 0] = class_idx

        # --- Guardar debug solo una vez ---
        #if not self.saved_debug_image:
        #    os.makedirs("debug_masks", exist_ok=True)
        #    # Guardar la imagen original en gris
        #    img_np = image_tensor.squeeze(0).numpy()
        #    plt.imsave(os.path.join("debug_masks", f"input_{name}"), img_np, cmap='gray')

            # Guardar la máscara combinada en color jet
        #    mask_np = combined_mask.numpy()
        #    plt.imsave(os.path.join("debug_masks", f"combined_mask_{name}"), mask_np, cmap='jet', vmin=0, vmax=len(self.masks_dirs))

        #    print(f"[DEBUG] Guardada imagen de entrada y máscara combinada en carpeta 'debug_masks' con nombre {name}")
        #    self.saved_debug_image = True

        return image_tensor, combined_mask

# --- Transformaciones ---
image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
mask_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
    transforms.PILToTensor()
])

# --- Cargar Datos ---
dataset = SegmentationDataset(
    images_dir=r"C:/Users/ezeem/OneDrive/Escritorio/Materias/ProcesamientoImagenes2/nuevo_data/dataset/images/train",
    masks_dirs=[
        r"C:/Users/ezeem/OneDrive/Escritorio/Materias/ProcesamientoImagenes2/nuevo_data/dataset/masks_lvendo/train",
        r"C:/Users/ezeem/OneDrive/Escritorio/Materias/ProcesamientoImagenes2/nuevo_data/dataset/masks_lvepi/train",
        r"C:/Users/ezeem/OneDrive/Escritorio/Materias/ProcesamientoImagenes2/nuevo_data/dataset/masks_rvendo/train"
    ],
    image_transform=image_transform,
    mask_transform=mask_transform
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- Inicialización ---
model = SegFormer(in_chans=1, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# OneCycleLR requiere steps_per_epoch > 0
steps_per_epoch = max(1, len(dataloader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()
torch.cuda.empty_cache()

# --- Métricas ---
def pixel_accuracy(pred, target, ignore_index=-100):
    pred_labels = torch.argmax(pred, dim=1)
    mask = target != ignore_index
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    correct = (pred_labels[mask] == target[mask]).sum()
    return correct.float() / mask.sum()

def dice_coeff(pred, target, num_classes=4, ignore_index=-100):
    pred_labels = torch.argmax(pred, dim=1)
    batch_dice = 0.0
    for class_idx in range(num_classes):
        pred_mask = (pred_labels == class_idx)
        true_mask = (target == class_idx)
        valid_mask = (target != ignore_index)
        intersection = (pred_mask & true_mask & valid_mask).float().sum()
        union = (pred_mask & valid_mask).float().sum() + (true_mask & valid_mask).float().sum()
        class_dice = (2. * intersection + 1e-6) / (union + 1e-6)
        batch_dice += class_dice
    return batch_dice / num_classes

def iou_score(pred, target, num_classes=4, ignore_index=-100):
    pred_labels = torch.argmax(pred, dim=1)
    batch_iou = 0.0
    for class_idx in range(num_classes):
        pred_mask = (pred_labels == class_idx)
        true_mask = (target == class_idx)
        valid_mask = (target != ignore_index)
        intersection = (pred_mask & true_mask & valid_mask).float().sum()
        union = (pred_mask | true_mask & valid_mask).float().sum()
        batch_iou += (intersection + 1e-6) / (union + 1e-6)
    return batch_iou / num_classes

# --- Entrenamiento ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for images, masks in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)  # [B, C, H_pred, W_pred]

            # 🔧 Ajustar tamaño del output al tamaño real de la máscara (202x202)
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode="bilinear", align_corners=False)

            loss = criterion(outputs, masks.long())


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            outputs_f32 = outputs.float()
            running_loss += loss.item()
            running_acc += pixel_accuracy(outputs_f32, masks, ignore_index=-100)
            running_dice += dice_coeff(outputs_f32, masks, num_classes=NUM_CLASSES, ignore_index=-100)
            running_iou += iou_score(outputs_f32, masks, num_classes=NUM_CLASSES, ignore_index=-100)

    scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Dice: {epoch_dice:.4f} | mIOU: {epoch_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Guardar checkpoint
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'metrics': {'accuracy': epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc, 'dice': epoch_dice, 'iou': epoch_iou}
        }, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), f'modelo_pesos_SegFormer_4C_epoch{epoch+1}.pth')
        torch.cuda.empty_cache()

# --- Evaluación final: guardar predicciones de todo el dataset ---
model.eval()
os.makedirs("predictions", exist_ok=True)
with torch.no_grad():
    for i, (img, mask) in enumerate(dataloader):
        img = img.to(DEVICE)
        with torch.cuda.amp.autocast():
            pred = model(img)
        pred_labels = torch.argmax(pred, dim=1).cpu().numpy()  # [B,H,W]
        masks_np = mask.numpy()
        for b in range(pred_labels.shape[0]):
            # guardar predicción coloreada (jet) y máscara original como PNG
            pred_map = pred_labels[b]
            true_map = masks_np[b]

            pred_path = os.path.join("predictions", f"pred_{i}_{b}.png")
            true_path = os.path.join("predictions", f"true_{i}_{b}.png")
            inp_path = os.path.join("predictions", f"input_{i}_{b}.png")

            plt.imsave(pred_path, pred_map, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
            plt.imsave(true_path, true_map, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)

            # guardar la imagen de entrada (desnormalizar visualmente)
            inp = img[b].cpu().squeeze(0).numpy()  # grayscale [H,W]
            inp = (inp * 0.5) + 0.5  # desnormalizar
            plt.imsave(inp_path, inp, cmap='gray')

print("Entrenamiento y guardado final completados. Predicciones en ./predictions/")

#curva de segmentation
