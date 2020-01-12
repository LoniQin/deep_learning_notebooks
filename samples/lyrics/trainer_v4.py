import utils
from tools.RNNLanguageManager import *
from samples.lyrics.common import read_lyrics
if __name__ == "__main__":
    filename = utils.get_file_name(__file__) + ".params"
    indices, index_to_char, char_to_index, _ = read_lyrics()
    parameter = RNNLanguageParameter(
        num_steps = 35,
        num_epochs=10,
        indices = indices,
        index_to_char = index_to_char,
        char_to_index = char_to_index,
        filename = filename,
        context=utils.try_gpu(),
        rnnType=rnn.LSTM
    )

    manager = RNNLanguageManager(parameter)
    manager.initialize()
    def per_epoch_finish_handler(param):
        print(param)
        index = random.randint(0, len(indices) - 2)
        prefix = [index_to_char[index] for index in indices[index: index + 2]]
        print(manager.predict(prefix, 50))
    parameter.per_epoch_finish_handler = per_epoch_finish_handler
    manager.train()