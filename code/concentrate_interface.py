from attr_interface import find_attr_vec
from word2vec_interface import find_word_vec


def find_concentrate_vec(word):
    try:
        attr_v = find_attr_vec(word)
        word_v = find_word_vec(word)
        concentrate_v = list(attr_v) + list(word_v)
        return concentrate_v
    except:
        return None


if __name__ == '__main__':
    print(find_concentrate_vec('goldfish'))
