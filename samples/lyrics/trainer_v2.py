import utils
from samples.lyrics.RNNLanguageManager import *
if __name__ == "__main__":
    filename = utils.get_file_name(__file__) + ".params"
    indices, index_to_char, char_to_index, _ = read_lyrics()
    parameter = RNNLanguageParameter(
        num_steps = 35,
        num_epochs=100,
        indices = indices,
        index_to_char = index_to_char,
        char_to_index = char_to_index,
        filename = filename,
        per_epoch_finish_handler=lambda x: print(x),
        finish_handler= lambda: print('Finish'),
        context=utils.try_gpu()
    )
    manager = RNNLanguageManager(parameter)
    manager.initialize()
    manager.finish_handler = lambda x: \
        print(x);print(manager.predict('你', 100))
    manager.train()