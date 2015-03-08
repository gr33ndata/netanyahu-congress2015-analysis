import nltk
import nltk.data
import nltk.corpus
from nltk.tokenize import word_tokenize

def file_id(filename):
    return filename.rsplit('.',1)[0].rsplit('_',1)[1]

def file_marker(remove_stopwords):
    return 'no_stopwords' if remove_stopwords else 'with_stopwords'

def read_file(filename=''):
    with open(filename, 'r') as fd:
        txt = fd.read()
    return txt

def dump_tuples(filename='', data=[], header=()):
    with open(filename, 'w') as fd:
        if header:
            placeholders = ','.join(['%s' for item in header]) + '\n'
            fd.write(placeholders % header)
        for data_tuple in data:
            placeholders = ','.join(['%s' for item in data_tuple]) + '\n'
            fd.write(placeholders % data_tuple)

def sentences(txt=''):
    txt = txt.strip()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenz = sent_detector.tokenize(txt)
    return tokenz 

def words(txt=''):
    return word_tokenize(txt)

def normalize(tokz=[], min_len=2, remove_stopwords=True):
    if remove_stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(["--", "'s", "n't", "'ve", "'re", "'ll", "and"])
    else:
        stopwords = []
    tokz = [
        tok.lower().replace('.','').replace(',','')
        for tok in tokz
        if len(tok) >= min_len
    ]
    tokz = [
        tok 
        for tok in tokz
        if not tok in stopwords 
    ]
    return tokz

def freq(tokz=[]):
    fdict = {}
    for tok in tokz:
        fdict[tok] = fdict.get(tok, 0) + 1
    return fdict

def top_freq(fdict={}, n=10):
    flist = fdict.items()
    flist.sort(key=lambda x: x[1], reverse=True)
    if n:
        flist = flist[0:n]
    return flist

def collocations(tokz):
    word_list = ['iran', 'israel', 'america', 'deal', 'jewish', 'islam']
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    #bigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(
        tokz,
        window_size = 3
    )
    finder.apply_freq_filter(3)
    def my_bigram_filter(w1, w2):
        if w1 in word_list or w2 in word_list:
            return False
        else: 
            return True
    finder.apply_ngram_filter(my_bigram_filter)
    collz = finder.nbest(bigram_measures.pmi, 30)
    collz = [(a, b) if a in word_list else (b, a) for a, b in collz] 
    return collz

def analyze_files(filenames=[], remove_stopwords=True):
    text = ''
    for filename in filenames:
        text += read_file(filename)
    tokz  = words(text)
    print 'Word Count: {0}'.format(len(tokz))
    tokz  = normalize(tokz, remove_stopwords=remove_stopwords)
    print 'Normalized Word Count: {0}'.format(len(tokz))
    fdict = freq(tokz)
    print 'Vocabulary: {0}'.format(len(fdict))
    freqz = top_freq(fdict, n=0)
    dump_tuples(
        filename='netanyahu_all.csv', 
        data=freqz, 
        header=('term', 'count')
    )

def analyze_file(filename='', remove_stopwords=True):
    fileId = file_id(filename)
    fileMarker = file_marker(remove_stopwords)
    text  = read_file(filename)
    tokz  = words(text)
    print 'Word Count: {0}'.format(len(tokz))
    tokz  = normalize(tokz, remove_stopwords=remove_stopwords)
    print 'Normalized Word Count: {0}'.format(len(tokz))
    coll  = collocations(tokz)
    dump_tuples(
        filename='netanyahu_collocations_%s.csv' % fileId, 
        data=coll, 
        header=('term1', 'term2')
    )
    fdict = freq(tokz)
    print 'Vocabulary: {0}'.format(len(fdict))
    freqz = top_freq(fdict, n=0)
    dump_tuples(
        filename='netanyahu_freqs_%s_%s.csv' % (fileMarker, fileId), 
        data=freqz, 
        header=('term', 'count')
    )


if __name__ == '__main__':

    analyze_files(
        filenames=[
            'netanyahu_2015.txt', 
            'netanyahu_2011.txt', 
            'netanyahu_1996.txt'
        ], 
        remove_stopwords=True
    )

    analyze_file(
        filename='netanyahu_2015.txt', 
        remove_stopwords=True
    )
