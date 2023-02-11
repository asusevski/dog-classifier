from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric, load_dataset
import numpy as np
from torchvision.transforms import Compose, ColorJitter, ToTensor, RandomPerspective
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoModelForImageClassification


dataset = load_dataset('imagefolder', data_dir="./data/dogs")


model_name = "google/vit-base-patch16-224"


feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


augs = Compose(
    [
        ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4),
        RandomPerspective(distortion_scale=0.3),
        ToTensor(),
    ]
)

def preprocess_images(examples):
  images = examples['image']  
  images = [np.array(augs(image.convert('RGB'))) for image in images]
  inputs = feature_extractor(images=images) 
  examples['pixel_values'] = inputs['pixel_values']
  return examples


train_dataset = dataset['train'].map(preprocess_images, batched=True)
validation_dataset = dataset['validation'].map(preprocess_images, batched=True)
test_dataset = dataset['test'].map(preprocess_images, batched=True)


labels = train_dataset.features['label'].names


# HF AutoModel args
num_labels = len(labels)
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}


model = AutoModelForImageClassification.from_pretrained(
  model_name,
  num_labels=num_labels,
  id2label=id2label,
  label2id=label2id,
  ignore_mismatched_sizes=True
)


batch_size = 8
logging_steps = int(len(train_dataset) // batch_size)
output_dir = "./models/hf_trainer"


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_steps=logging_steps,
    push_to_hub=True
)


metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)


trainer.train()
