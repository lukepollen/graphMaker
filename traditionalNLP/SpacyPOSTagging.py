import pandas as pd
import spacy


# Return a list that gives us a sentence broken in to its parts of speech
def posTags(sentence):
    tokens = nlp(sentence)
    tags = []
    for tok in tokens:
        tags.append(tok.tag_)

    return tags


# Removes sequences that start with VB, but are not an exact match of VBN, NN.
def booleanStartsVBX(posList):
    # Check whether the start of sequences has VB format, and flag for removal if not exactly VBN NN format.
    if posList[0].startswith("VB"):
        # Keep 'VBN, NN' formats
        if len(posList) == 2 and posList[0] == 'VBN' and posList[1] == 'NN':
            return False
        return True
    else:
        return False


# Flags whether the PoS list started with an _SP Flag
def startsWithSp(posList):
    if posList[0].startswith('_SP'):
        return True
    else:
        return False


# Flags whether the PoS list started with a : Flag
def startsWithCo(posList):
    if posList[0].startswith(':'):
        return True
    else:
        return False


# Flags whether the PoS list started with an RB Flag
def startsWithRb(posList):
    if posList[0].startswith('RB'):
        return True
    else:
        return False


# Flag rows with artifact '€'
def hadBadChar(posList):
    anyText = ' '.join(posList)
    if '€' in anyText:
        return True
    else:
        return False


# Brute force approach to ensure we only have sequences that contain a noun ending.
def endsWithNNX(posList):
    # Skip to end of list. Inspect element. If NN type, return true.
    # Uses startswith in last element because final element may be NN, or NNX format.
    if posList[-1].startswith('NN'):
        return True
    else:
        return False


# Checks whether we have an agreeable ratio of quantative symbols to descriptive - 1:3 or less. Removes gibberish quant strings.
def isBadCDRatio(posList):
    # Take occurences of 'CD' parts in list of parts, and multipy by three.
    totalLength = len(posList)
    # print(totalLength)
    cdCount = posList.count('CD')
    # print(cdCount)
    maxValidLength = cdCount * 3
    # print(maxValidLength)

    # Sequence is valid if CD occurencesa * 3 is less than the total length of the list of speech elements.
    if maxValidLength > totalLength:
        return True
    else:
        return False


# Check whether the word sequence was a valid string
def isBadLength(posList):
    concatnated = ''.join(posList)
    if len(concatnated) < 4:
        return True
    else:
        return False


# Take the previously computed symbolic/numeric to alpha and include qualitiative + noun where these are within the Page URL.
def finalCDCheck(dataFrame):
    ## Check three truth series; whether flagged as bad, whether quantative + descriptive, whether in url.
    ## We need to reverse the truth value  if all true to exclude these rows from being filtered.

    # Check whether we have some quantitative plus object part of speech in row
    cdnn = [True if x == ['CD', 'NN'] else True if x == ['CD', 'NNS'] else False for x in
            dataFrame['Parts of Speech']]
    cdnnSeries = pd.Series(cdnn)

    # Truth values whether text was considered potentially indescriptive.
    isBadRatio = dataFrame['isBadCDRatio']

    # Check whether the URL actually contained the suspicious looking text.
    # wordsInURL = dataFrame['Word Sequence'].isin(dataFrame['Page'])
    wordSequences = list(dataFrame['Word Sequence'])
    theURLs = list(dataFrame['Page'])
    # wordsInURL = pd.Series([True if x.replace(" ", "-") in theURLs else False for x in wordSequences])
    # wordsInURL = pd.Series([True if x.replace(" ", "-") in  theURLs else False for x in wordSequences])
    zippedSequencePair = zip(wordSequences, theURLs)
    wordsInURL = pd.Series([True if x.replace(" ", "-") in y else False for x, y in zippedSequencePair])

    # Leave bad CD ratio as True, for removal, if bad, and not a CD, NN sequence in the Page URL.
    # dataFrame['FinalCDTruth'] = dataFrame[ (dataFrame['isBadCDRatio'])  \
    #                             #& ( (dataFrame['Parts of Speech'] == ['CD', 'NN']) \
    #                             #or (dataFrame['Parts of Speech'] == ['CD', 'NNS']) ) \
    #                             & pd.Series(cdnn) \
    #                             & (dataFrame['Word Sequence'].isin(dataFrame['Page']))
    #                             ]

    # Now we've identified which elements which have a bad ratio, but are of the right form and are also in url
    wasActuallyGood = isBadRatio & cdnnSeries & wordsInURL

    # Invert truth, so has False when evaluating for removal, because these terms were actually good.
    doNotRemove = ~wasActuallyGood

    # The True value, for original removal, will now collide with False, having set False where all three conditions were True.
    # True(original judgement) + False(Inverse of all exception conditions) = False. Do not remove.
    # True(original judgement) + True(Was not an exception) = True. Still flagged to remove.
    dataFrame['finalCDTruth'] = isBadRatio & doNotRemove

    return dataFrame


nlp = spacy.load('en_core_web_sm')