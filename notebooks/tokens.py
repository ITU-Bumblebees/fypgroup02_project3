import Config

with open(Config.hate_train_text_path, 'r', encoding = 'utf-8') as infile:
    train_hate = [line.strip() for line in infile]

with open(Config.abortion_train_text_path, 'r', encoding = 'utf-8') as infile:
    train_abortion = [line.strip() for line in infile]


for line in train_hate:
    print(line)