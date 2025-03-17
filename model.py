from roboflow import Roboflow
from torch.utils.data import Dataset
import os
from PIL import Image
from transformers import SegformerFeatureExtractor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from evaluate import load
import torch
from torch import nn
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor

        # Читаем _classes.csv
        self.classes_csv_file = os.path.join(self.root_dir, "_classes.csv")
        with open(self.classes_csv_file, 'r') as fid:
            lines = fid.readlines()
            data = [line.strip().split(',') for line in lines[1:]]  # Пропускаем заголовок
            self.id2label = {str(id_): name for id_, name in data}

        # Собираем файлы
        image_file_names = [f for f in os.listdir(self.root_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        mask_file_names = [f for f in os.listdir(self.root_dir) if f.lower().endswith('.png')]

        # Извлекаем базовые имена, убирая суффикс _mask из масок
        image_bases = {f.replace('.jpg', '').replace('.jpeg', '') for f in image_file_names}
        mask_bases = {f.replace('_mask.png', '').replace('.png', '') for f in mask_file_names}

        # Находим общие базовые имена
        common_bases = sorted(image_bases & mask_bases)
        self.images = [f"{base}.jpg" for base in common_bases]  # Предполагаем .jpg, можно добавить .jpeg
        self.masks = [f"{base}_mask.png" for base in common_bases]  # Добавляем _mask для масок

        print(f"Images: {len(self.images)}, Masks: {len(self.masks)}")
        print(f"First 5 images: {self.images[:5]}")
        print(f"First 5 masks: {self.masks[:5]}")
        print(f"id2label: {self.id2label}")
        if not self.images or not self.masks:
            raise ValueError(f"Нет парного соответствия изображений и масок в {self.root_dir}")
        if len(self.images) != len(self.masks):
            raise ValueError("Количество изображений и масок не совпадает!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.root_dir, self.masks[idx]))
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs


class SegformerFinetuner(pl.LightningModule):

    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.train_mean_iou = load("mean_iou")
        self.val_mean_iou = load("mean_iou")
        self.test_mean_iou = load("mean_iou")
        self.validation_step_outputs = []
        print("Model device:", next(self.model.parameters()).device)

    def forward(self, images, masks=None):
        images = images.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        # print("\nImages device:", images.device)
        # print("Masks device:", masks.device if masks is not None else "None")
        outputs = self.model(pixel_values=images, labels=masks)
        return outputs

    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            for k, v in metrics.items():
                self.log(k, v)
            return metrics
        else:
            return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        self.validation_step_outputs.append({'val_loss': loss})
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )
        avg_val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        metrics = {"val_loss": avg_val_loss, "val_mean_iou": val_mean_iou, "val_mean_accuracy": val_mean_accuracy}
        for k, v in metrics.items():
            self.log(k, v)
        self.validation_step_outputs.clear()

    # Оставляем test_step и test_epoch_end как есть, но для единообразия можно обновить и их
    def test_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]
        metrics = {"test_loss": avg_test_loss, "test_mean_iou": test_mean_iou, "test_mean_accuracy": test_mean_accuracy}
        for k, v in metrics.items():
            self.log(k, v)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("PyTorch version:", torch.__version__)
    print("Lightning version:", pl.__version__)

    rf = Roboflow(api_key="q3YSMGxcnMRqHCb9ppWg")
    project = rf.workspace("uit-kbay3").project("screen-segmentation")
    version = project.version(3)
    dataset = version.download("png-mask-semantic")

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.reduce_labels = False
    feature_extractor.size = 128

    train_dataset = SemanticSegmentationDataset("./Screen-segmentation-3/train/", feature_extractor)
    val_dataset = SemanticSegmentationDataset("./Screen-segmentation-3/test/", feature_extractor)
    test_dataset = SemanticSegmentationDataset("./Screen-segmentation-3/test/", feature_extractor)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, prefetch_factor=None)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, prefetch_factor=None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, prefetch_factor=None)

    # print(f"Images: {len(train_dataset.images)}, Masks: {len(train_dataset.masks)}")
    # print(f"First 5 images: {train_dataset.images[:5]}")
    # print(f"First 5 masks: {train_dataset.masks[:5]}")

    segformer_finetuner = SegformerFinetuner(
        train_dataset.id2label,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        metrics_interval=10,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

    trainer = pl.Trainer(
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=100,
        val_check_interval=len(train_dataloader),
        accelerator="gpu",
        devices=1
        # precision=16  # Включаем FP16
    )

    print("Trainer device:", trainer.device_ids)
    trainer.fit(segformer_finetuner)
    print("Model device after training:", next(segformer_finetuner.model.parameters()).device)