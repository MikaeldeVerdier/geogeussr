import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class TorchDataset(Dataset):
    def __init__(self, annotations, preprocessor, region_names=None, data_augmentor=None, use_augmentation=False, rets=[], processor_kwargs={}):
        self.annotations = self.get_region_annotations(annotations, region_names)
        self.preprocessor = preprocessor
        self.data_augmentor = data_augmentor
        self.use_augmentation = use_augmentation
        self.rets = rets
        self.processor_kwargs = processor_kwargs

    def get_region_annotations(self, annotations, region_names):
        if region_names is None:
            return annotations

        region_annotations = [
            annotation
            for annotation in annotations
            if annotation["location"]["coding"]["country"] in region_names
        ]

        return region_annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        result = self.preprocessor([annotation], use_augmentation=self.use_augmentation, rets=self.rets, **self.processor_kwargs)[0]

        return result

    def collate_fn(self, batch):
        image_inputs = torch.stack([torch.from_numpy(sample["pixel_values"]).squeeze(0) for sample in batch])
        input_ids = [torch.from_numpy(t["input_ids"]).squeeze(0) for t in batch]
        attention_mask = [torch.from_numpy(t["attention_mask"]).squeeze(0) for t in batch]

        pad_token = self.preprocessor.clip_processor.tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        collated_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image_inputs
        }

        return collated_inputs

    def get_loader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, num_workers=num_workers)
