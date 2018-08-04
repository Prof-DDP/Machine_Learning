# Using a markov chains to generate text

__author__ = 'Prof.D'

import random

# For text like sentences, paragraphs, articles, etc.
'''
text = 'peter piper picked a peck of pickled peppers.'
'''
def markov(txt, order, n_times):
    ngrams = {}

    for i in range((len(txt) - order)):
        gram = txt[i:(i+order)]
        '''
        print(gram)
        '''
        if gram not in ngrams.keys():
            ngrams[gram] = []
        
        ngrams[gram].append(txt[i + order])

    ngrams[txt[-order:]] = ''

    current = txt[:order]
    result = current
    max_times = len(txt)
    
    if n_times <= max_times:
        for i in range(n_times):
            possibilities = ngrams[current]
            if not possibilities:
                break
            next_ = random.choice(possibilities)
            result += next_
            length = len(result)
            current = result[length-order:length] #Had txt here instead of result. Caused a lot of issues later on.
        return result
    else:
        print('Max number of times: ', max_times)
        
# For name and title generation
def markov_generator(source_txt, order, n_times):
    ngrams = {}
    beginnings = []
    grams = []
    for j in range(len(source_txt)):
        txt = source_txt[j]
        for i in range((len(txt)-order)):
            gram = txt[i : (i + order)]
            grams.append(gram)
            if i == 0:
                beginnings.append(gram)
            if gram not in ngrams.keys():
                ngrams[gram] = []
                
            ngrams[gram].append(txt[i + order])
        ngrams[txt[-order:]] = ['']
    
    ngrams[txt[-order:]] = ['']
    
    current = random.choice(beginnings)
    result = current
    for i in range(n_times):
        possibilites = ngrams[current]
        if not possibilites:
            break
        next_ = random.choice(possibilites)
        result += next_
        length = len(result)
        current = result[(length - order): length]
    
    return result

'''
name = markov_generator(names, 4, 100)
'''










