import pke

class KeyPhraseExtractor:
    def __init__(self, language = 'en', numKeyPhrases = 10):
        self.language = language
        self.numKeyPhrases = numKeyPhrases


    def getKeyPhrases(self, text_file_path):
        # initialize keyphrase extraction model, here TopicRank
        extractor = pke.unsupervised.TopicRank()

        # load the content of the document, here document is expected to be in raw
        # format (i.e. a simple text file) and preprocessing is carried out using spacy
        extractor.load_document(input=text_file_path, language=self.language)

        # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
        # and adjectives (i.e. `(Noun|Adj)*`)
        extractor.candidate_selection()

        # candidate weighting, in the case of TopicRank: using a random walk algorithm
        extractor.candidate_weighting()

        # N-best selection, keyphrases contains the 10 highest scored candidates as
        # (keyphrase, score) tuples
        keyphrases = extractor.get_n_best(n=self.numKeyPhrases)

        return [keyphrase for (keyphrase, score) in keyphrases]

    def getKeyPhraseSentencesSimilarity(self, text_file_path, sentences):
        keyphrase_scores = []
        keyphrases = self.getKeyPhrases(text_file_path)
        for sentence in sentences:
            count = 0.0
            for keyphrase in keyphrases:
                if keyphrase in sentence:
                    count += 1
            keyphrase_scores.append(count**2)
        return keyphrase_scores
