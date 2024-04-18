from django.shortcuts import render
import RAKE
from collections import OrderedDict
from sense2vec import Sense2Vec
# Step 1: Extractive Text Summarization using Bert Extractive Summarizer
from summarizer import Summarizer

def extractive_summarization(text):
    model = Summarizer()
    result = model(text, min_length=60, max_length=500, ratio=0.4)
    summarized_text = ''.join(result)
    return summarized_text

# Step 2: Keyword Extraction using RAKE

def keyword_extraction(text):
    stop_dir = "SmartStoplist.txt"
    rake_object = RAKE.Rake(stop_dir)
    keywords = rake_object.run(text)
    return keywords

# Step 3: Distractor Generation using Sense2Vec


def distractor_generation(word, s2v):
    output = []
    word = word.lower().replace(" ", "_")
    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=20)
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ").lower()
        if append_word.lower() != word:
            output.append(append_word.title())
    out = list(OrderedDict.fromkeys(output))
    return out

# Step 4: Question Generation using fine-tuned T5 transformer

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def question_generation(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    with torch.no_grad():
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            early_stopping=True,
            num_beams=5,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            max_length=72
        )
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    question = dec[0].replace("question:", "").strip()
    return question

# Fine-tuning T5 transformer using your own dataset of MCQ questions
def fine_tune_t5(dataset_path, model_save_path):
    # Load your dataset
    dataset = pd.read_csv(dataset_path)

    # Define your training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Initialize T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Define a function to preprocess your dataset
    def preprocess_function(examples):
        inputs = tokenizer(
            examples['question'] + " options: " + examples['options'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        labels = tokenizer(examples['answer'], padding='max_length', truncation=True, max_length=512, return_tensors='pt').input_ids

        return inputs, labels

    # Preprocess your dataset
    processed_dataset = dataset.map(preprocess_function, batched=True)

    # Define your trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=None, 
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(model_save_path)

fine_tune_t5("dataset.csv", "fine_tuned_t5_model")



summarized_text = extractive_summarization(text)
print("Summarized Text:", summarized_text)

keywords = keyword_extraction(text)
print("Keywords:", keywords)

# Initialize Sense2Vec model
s2v = Sense2Vec().from_disk("path_to_s2v_model")

word = "Natural Language Processing"
distractors = distractor_generation(word, s2v)
print("Distractors for", word, ":", distractors)

# Initialize fine-tuned T5 model and tokenizer
model_path = "path_to_fine_tuned_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


question = question_generation(context, answer, model, tokenizer)
print("Generated Question:", question)

