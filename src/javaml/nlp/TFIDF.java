package javaml.nlp;

import java.util.HashMap;
import java.util.Map;

public class TFIDF {

    public double calculateTF(String[] document, String term) {
        int termFrequency = 0;
        for (String word : document) {
            if (word.equalsIgnoreCase(term)) {
                termFrequency++;
            }
        }
        return (double) termFrequency / document.length;
    }

    public double calculateIDF(String[][] corpus, String term) {
        int documentCount = 0;
        for (String[] document : corpus) {
            for (String word : document) {
                if (word.equalsIgnoreCase(term)) {
                    documentCount++;
                    break;
                }
            }
        }
        return Math.log((double) corpus.length / (documentCount + 1));
    }

    public Map<String, Double> calculateTFIDF(String[] document, String[][] corpus) {
        Map<String, Double> tfidfMap = new HashMap<>();
        for (String term : document) {
            double tf = calculateTF(document, term);
            double idf = calculateIDF(corpus, term);
            tfidfMap.put(term, tf * idf);
        }
        return tfidfMap;
    }
}
