import pandas as pd
import numpy as np

class DataLoader:

    @staticmethod
    def load_validation_data():
        # load validation data
        val_data = pd.read_csv("../data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv")
        print("Labels: ", val_data.columns.tolist())
        val_right_ending_nr = val_data[['AnswerRightEnding']].values
        val_context_sentences = val_data.iloc[:, 1:5].values
        val_ending_sentence1 = val_data[['RandomFifthSentenceQuiz1']].values
        val_ending_sentence2 = val_data[['RandomFifthSentenceQuiz2']].values
        return val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2

    @staticmethod
    def load_test_data_with_right_ending_nr():
        # load test data (for experimental section in report, containing right ending nr)
        val_data = pd.read_csv("../data/test_for_report-stories_labels.csv")
        print("Labels: ", val_data.columns.tolist())
        val_right_ending_nr = val_data[['AnswerRightEnding']].values
        val_context_sentences = val_data.iloc[:, 1:5].values
        val_ending_sentence1 = val_data[['RandomFifthSentenceQuiz1']].values
        val_ending_sentence2 = val_data[['RandomFifthSentenceQuiz2']].values
        return val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2

    @staticmethod
    def load_test_data_to_make_predictions():
        # load test data (to make the predictions that we need to hand in as a .csv)
        val_data = pd.read_csv("../data/test-stories.csv")
        print("Labels: ", val_data.columns.tolist())
        val_context_sentences = val_data.iloc[:, 0:4].values
        val_ending_sentence1 = val_data[['RandomFifthSentenceQuiz1']].values
        val_ending_sentence2 = val_data[['RandomFifthSentenceQuiz2']].values
        return val_context_sentences, val_ending_sentence1, val_ending_sentence2

    @staticmethod
    def load_training_data():
        # load training data
        train_data = pd.read_csv("../data/train_stories.csv")
        print("Training data: ", train_data.head())
        print("Labels: ", train_data.columns.tolist())
        train_context_sentences = train_data.iloc[:, 2:6].values
        train_ending_sentence = train_data[['sentence5']].values
        train_story_title = train_data[['storytitle']].values
        return train_context_sentences, train_ending_sentence, train_story_title

    def load_data_with_fake_endings(self, file_name):
        # should return in the same format as load validation data, but with fake endings as ending2
        train_context_sentences, train_ending_sentence, _ = self.load_training_data()
        counter = 0
        discarded = 0
        sentences = open(file_name, "r")  # opens file for reading
        fake_endings = []
        new_train_context_sentences, new_train_ending_sentence = [], []
        for sentence in sentences:
            if len(sentence) < 10:
                # don't include the sample in the new training data
                discarded += 1
            else:
                # append data
                new_train_context_sentences.append(train_context_sentences[counter])
                new_train_ending_sentence.append(train_ending_sentence[counter])
                fake_endings.append(sentence)
            counter += 1

        fake_endings = np.array(list(map(lambda i: i[:-1], fake_endings)))
        fake_endings = fake_endings.reshape(len(fake_endings), 1)
        right_endings = np.ones(len(fake_endings))
        return right_endings, np.array(new_train_context_sentences), np.array(new_train_ending_sentence), fake_endings