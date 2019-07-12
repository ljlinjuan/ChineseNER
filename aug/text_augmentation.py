# Do text augmentation with Nouns and Verbs. (同义词转换)
# By LinJuan 20190625
# ref: kaggle.com/init927/nlp-data-augmentation

import re
import spacy
import random
from thesaurus import Word
import nltk
from nltk.corpus import wordnet
import en_core_web_sm
import warnings


warnings.filterwarnings('ignore')
class DataAugmentation(object):
    
    def __init__(self,text,percent=50):
        self.text = text
        self.percent = percent
        
    def synalter_Noun_Verb(self, word, al, POS):
        max_temp = -1
        flag = 0
        for i in al:
            try:
                w1 = wordnet.synset(word + '.' + POS + '.01')
                w2 = wordnet.synset(i + '.' + POS + '.01')  # n denotes noun
                if (max_temp < w1.wup_similarity(w2)):
                    max_temp = w1.wup_similarity(w2)
                    temp_name = i
                    flag = 1
            except:
                f = 0

        if flag == 0:
            max1 = -1.
            nlp = en_core_web_sm.load()
            for i in al:
                j = i.replace(' ', '')
                tokens = nlp(u'' + j)
                token_main = nlp(u'' + word)
                for token1 in token_main:
                    if max1 < float(token1.similarity(tokens)):
                        max1 = token1.similarity(tokens)
                        value = i
            max1 = -1.
            return value
        else:
            return temp_name
    
    def generate_output_text(self):
        text = self.text
        output_text = text
#         print(output_text)
        words = text.split()
        counts = {}
        for word in words:
            if word not in counts:
                counts[word] = 0
            counts[word] += 1
        
        one_word = []
        for key, value in counts.items():
            if value == 1 and key.isalpha() and len(key) > 2:
                one_word.append(key)
        
        noun = []
        verb = []
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(u'' + ' '.join(one_word))
        for token in doc:
            if token.pos_ == 'VERB':
                verb.append(token.text)
            if token.pos_ == 'NOUN':
                noun.append(token.text)
                
#         print('\n verb: \n')
#         print(str(len(verb)))
#         print(verb)
        
#         print('\n noun: \n')
#         print(str(len(noun)))
#         print(noun)
        
        all_main = verb + noun
        len_all = len(noun) + len(verb)
        final_value = int(len_all * self.percent / 100)
        random.seed(4)
        temp = random.sample(range(0, len_all), final_value)
        for i in temp:
            try:
                word_str = all_main[i]
                w = Word(word_str)
                a1 = list(w.synonyms())
        
#                 print('\n'+str(i)+'\n')
#                 print(word_str)
#                 print(a1)
        
                if i < len(verb):
                    change_word = self.synalter_Noun_Verb(word_str, a1, 'v')
                    try:
                        search_word = re.search(r'\b(' + word_str + r')\b', output_text)
                        Loc = search_word.start()
                        output_text = output_text[:int(Loc)] + change_word + output_text[int(Loc) + len(word_str):]
                    except:
                        f = 0

                else:
                    change_word = self.synalter_Noun_Verb(word_str, a1, 'n')
                    try:
                        search_word = re.search(r'\b(' + word_str + r')\b', output_text)
                        Loc = search_word.start()
                        output_text = output_text[:int(Loc)] + change_word + output_text[int(Loc) + len(word_str):]
                    except:
                        f = 0

            except:
                f = 0
                
        return output_text


def main():
    # use one example for testing
    text="The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news,\
            and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent \
            publication and does not have a faculty advisor or any editorial oversight from the University."
    da = DataAugmentation(text)
    print(da.generate_output_text())
    
#     # --------a simple test--------
#     word_str='hope'
#     w = Word(word_str)
#     a1 = list(w.synonyms())
#     print(da.synalter_Noun_Verb(word_str, a1, 'v'))

    
if __name__ == "__main__":
    main()

