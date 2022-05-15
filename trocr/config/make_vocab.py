
with open('vocab.txt', 'r') as f_r:
    with open('my_vocab.txt', 'w') as f_w:
        for line in f_r:
                line = line.replace('\n', '')
                f_w.write(line + ' ' + str(1) + '\n')