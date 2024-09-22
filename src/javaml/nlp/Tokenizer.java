package javaml.nlp;

import java.util.ArrayList;
import java.util.List;

public class Tokenizer {

    public static final String PUNCTUATION_REGEX = "[.,!?\\-]";
    private List<String> stopWords;

    public Tokenizer(List<String> stopWords) {
        this.stopWords = stopWords;
    }

    public List<String> tokenize(String text) {
        text = text.toLowerCase();
        text = text.replaceAll(PUNCTUATION_REGEX, "");

        String[] tokens = text.split("\\s+");
        List<String> filteredTokens = new ArrayList<>();

        for (String token : tokens) {
            if (!stopWords.contains(token)) {
                filteredTokens.add(token);
            }
        }
        return filteredTokens;
    }

}
