import fasttext
fasttext.FastText.eprint = lambda x: None

def train_fasttext_model(train_data_path, model_output_path):
    model = fasttext.train_supervised(
        train_data_path, label_prefix="__label__", epoch=50, lr=1, wordNgrams=2, bucket=200000, dim=50, loss='ova')

    model.save_model(model_output_path)

def classify_text(model_path, text):
    model = fasttext.load_model(model_path)
    label, probability = model.predict(text, k=3)
    print("existing label array ", label)
    label = label[0].replace('__label__', '')

    return label, probability[0]

if __name__ == "__main__":
    train_data_path = "data.txt"
    model_output_path = "fasttext_model.bin"

    train_fasttext_model(train_data_path, model_output_path)

    input_text = "inside data single set will engineering  he  household find"

    label, probability = classify_text(model_output_path, input_text)

    print(f"Predicted result: {label}, Probability: {probability}")
