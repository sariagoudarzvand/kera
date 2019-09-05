
def predicting_topics(self):
    from keras.models import load_model
    import pandas as pd
    for epoch in range(1, self.NUM_EPOCHS + 1, 19):
        file_name = "./checkpoint/" + self.dir + str(epoch) + ".hdf5"
        lstm_autoencoder = load_model(file_name,
                                      {'sent_wids': self.sent_wids, 'score_cooccurance': self.score_cooccurance})
        encoder = Model(lstm_autoencoder.input, lstm_autoencoder.get_layer('modified_layer').output)
        # lstm_autoencoder = load_model(file_name, {'sent_wids': sent_wids})
        # encoder = Model(lstm_autoencoder.input, lstm_autoencoder.get_layer('encoder_lstm').output)
        topics = []
        weights = encoder.get_weights()[0]
        # print(weights)
        # layer1 = encoder.layers[0].output
        score = self.calc_pairwise_dev(weights)
        # np.save('test_ent.npy', layer1)
        for idx in range(encoder.output_shape[1]):
            token_idx = np.argsort(weights[:, idx])[::-1]
            topics.append(
                [(epoch, idx, self.id2word[x], weights[x, idx], score) for x in token_idx if x in self.id2word])

        for topic in topics:
            temp_df = pd.DataFrame(topic, columns=['epoch', 'index', 'word', 'weight', 'score'])
            with open('./csvfiles/' + self.dir + 'lstm_ae_with_layer_and_kate9.csv', 'a') as f:
                temp_df.to_csv(f, index=False)
        print(epoch)