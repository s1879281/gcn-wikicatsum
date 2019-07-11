from preprocessing import _normalize_text_cleaned
import os

src_path = 'fairseq/data/film_tok_min5_L7.5k'
tgt_path = 'fairseq/data/film_tok_min5_L7.5k/modified'

splits = ['train', 'valid', 'test']

def modify(src_path, tgt_path, split):
    def create_path(path, split):
        return os.path.join(path, split + '.src')
    with open(create_path(src_path, split), 'rb') as f1, open(create_path(tgt_path, split), 'a+', encoding='utf8') as f2:
        for line in f1.readlines():
            line = _normalize_text_cleaned(line.decode())
            f2.write(line + '\n')


if __name__ == "__main__":
    for split in splits:
        modify(src_path, tgt_path, split)
        print(split + ' finished.')

