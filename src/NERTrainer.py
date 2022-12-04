from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification

TOKENIZER = "dslim/bert-base-NER"

class NERTrainer:
    def __init__(self, batch_size, epochs, model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model

    def load_wikiann(self,n_samples_train="", n_samples_val="", n_samples_test=""):
        """
        load wikiann dataset with custom number of rows for each section
        """
        name = 'wikiann'
        lang = 'en'
        train, test, validation = load_dataset(name, lang, split=['train[:'+str(n_samples_train)+']','test[:'+str(n_samples_test)+']','validation[:'+str(n_samples_val)+']'])
        dataset = DatasetDict({'validation': validation,'test':test,'train':train})
        label_names = dataset["train"].features["ner_tags"].feature.names
        self.dataset = dataset.map(self.tokenize_adjust_labels, label_names, batched=True)

    def tokenize_adjust_labels(self,samples,label_names):
        tokenized_samples = self.tokenizer.batch_encode_plus(samples["tokens"], is_split_into_words=True, truncation=True)
        total_adjusted_labels = []
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = samples["ner_tags"][k]
            i = -1
            adjusted_label_ids = []

            for word_idx in word_ids_list:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function.
                if(word_idx is None):
                    adjusted_label_ids.append(-100)
                elif(word_idx!=prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = word_idx
                else:
                    label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(existing_label_ids[i])
            total_adjusted_labels.append(adjusted_label_ids)

        #add adjusted labels to the tokenized samples
        tokenized_samples["labels"] = total_adjusted_labels
        return tokenized_samples

    def train(self):
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        logging_steps = len(self.dataset['train']) // self.batch_size

        training_args = TrainingArguments(
            output_dir="results",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            disable_tqdm=False,
            logging_steps=logging_steps
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.log_history = trainer.state.log_history