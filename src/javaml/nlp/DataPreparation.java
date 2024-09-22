package javaml.nlp;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

public class DataPreparation {
    private Tokenizer tokenizer;

    public DataPreparation(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    public List<TrainingSample> prepareData(String text, int seqLen) {
        List<String> tokens = tokenizer.tokenize(text);
        List<TrainingSample> samples = new ArrayList<>();

        for (int i = 0; i < tokens.size() - seqLen; i++) {
            List<String> inputSeq = tokens.subList(i, i + seqLen);
            String target = tokens.get(i + seqLen);
            samples.add(new TrainingSample(inputSeq, target));
        }

        return samples;
    }

    public class TrainingSample {
        public List<String> inputSeq;
        public String target;

        public TrainingSample(List<String> inputSeq, String target) {
            this.inputSeq = inputSeq;
            this.target = target;
        }
    }
}
