# Itamar Trainin 315425967

import torch
import torch.nn as nn
import settings


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab, output_vocab, embd_dim, hidden_dim, output_dim, num_layers=1, attention=False):
        super(Seq2Seq, self).__init__()

        self.num_layers = num_layers
        self.attention = attention

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.encoder = Encoder(len(input_vocab), embd_dim, hidden_dim, num_layers)
        self.decoder = Decoder(len(output_vocab), embd_dim, hidden_dim, self.output_dim, num_layers, attention)

        self.dropout = nn.Dropout()
        self.lin = nn.Linear(self.output_dim, len(output_vocab))
        self.lin_attention = nn.Linear(self.hidden_dim, len(output_vocab))

        self.softmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

        self.attention_layer = Attention(hidden_dim, self.output_dim)

    def forward_train(self, x, y_true):
        encoder_outputs, encoder_hidden = self.encoder.forward(x)
        hidden = self.decoder.init_hidden()
        context_vector = torch.zeros((1, self.hidden_dim))
        # hidden = encoder_hidden

        current_y = self.output_vocab[settings.SENT_BEG]
        result = [current_y]
        loss = 0.0
        counter = 0
        while current_y != self.output_vocab[settings.SENT_END]:
            if len(result) < len(x) - 1:
                scores, hidden, context_vector, _ = self.forward(encoder_outputs, context_vector, current_y, hidden)

                loss += self.nllloss(scores, y_true[counter].unsqueeze(0))

                current_y = y_true[counter]
                y_pred = torch.argmax(scores, dim=1)
                result.append(int(y_pred))
                counter += 1
            else:
                result.append(self.output_vocab[settings.SENT_END])
                break

        loss = loss / (len(result) - 1)

        return torch.tensor(result).long(), loss

    def forward_predict(self, x, y_true):

        encoder_outputs, encoder_hidden = self.encoder.forward(x)

        hidden = self.decoder.init_hidden()
        context_vector = torch.zeros((1, self.hidden_dim))
        # hidden = encoder_hidden

        current_y = self.output_vocab[settings.SENT_BEG]
        result = [current_y]
        loss = 0.0
        all_weights = []
        counter = 0
        while current_y != self.output_vocab[settings.SENT_END]:
            if len(result) < len(x) - 1:
                scores, hidden, context_vector, weights = self.forward(encoder_outputs, context_vector, current_y, hidden)

                loss += self.nllloss(scores, y_true[counter].unsqueeze(0))

                all_weights.append(weights)
                current_y = torch.argmax(scores, dim=1)
                result.append(int(current_y))
                counter += 1
            else:
                result.append(self.output_vocab[settings.SENT_END])
                break

        loss = loss / (len(result) - 1)

        return torch.tensor(result).long(), loss, all_weights

    def forward(self, encoder_outputs, context_vector, current_y, hidden):
        weights = None

        word = torch.tensor([current_y])

        if self.attention:
            decoder_output, hidden = self.decoder.forward(context_vector, word, hidden)

            context_vector, weights = self.attention_layer.forward(encoder_outputs, decoder_output)

            output = self.lin_attention(context_vector)
            output = self.softmax(output)
        else:
            context_vector = encoder_outputs[-1].unsqueeze(0)

            decoder_output, hidden = self.decoder.forward(context_vector, word, hidden)

            output = self.dropout(decoder_output)
            output = self.lin(output)

        scores = self.softmax(output)

        return scores, hidden, context_vector, weights


class Encoder(nn.Module):
    def __init__(self, vocab_dim, embd_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embd = nn.Embedding(vocab_dim, embd_dim)
        self.lstm = nn.LSTM(embd_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=settings.bidirectional)

    def forward(self, words):
        output = self.embd(words).unsqueeze(0)
        output, hidden = self.lstm(output)
        output = output.squeeze(0)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_dim, embd_dim, encoder_out_dim, hidden_dim, num_layers=1, attention=False):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.attention = attention

        self.hidden_dim = hidden_dim
        self.input_dim = embd_dim + encoder_out_dim

        self.embd = nn.Embedding(vocab_dim, embd_dim)
        self. lstm = nn.LSTM(self.input_dim,
                             hidden_dim,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=settings.bidirectional)

    def forward(self, context_vector, words, hidden):
        embds = self.embd(words)
        context_vector = context_vector.expand(words.size(0), -1)
        output = torch.cat((context_vector, embds), 1).unsqueeze(0)
        output, hidden = self.lstm(output, hidden)
        output = output.squeeze(0)
        return output, hidden

    def init_hidden(self):
        zeros = torch.zeros(1, self.num_layers, self.hidden_dim)
        return (zeros, zeros)


class Attention(nn.Module):
    def __init__(self, encoder_output_dim, decoder_output_dim):
        super(Attention, self).__init__()

        self.W_a = nn.Linear(encoder_output_dim + decoder_output_dim, decoder_output_dim)
        self.v_a = nn.Linear(decoder_output_dim, 1)
        self.W_c = nn.Linear(encoder_output_dim + decoder_output_dim, encoder_output_dim)

    def forward(self, encoder_outputs, decoder_output):
        decoder_outputs = decoder_output.expand(encoder_outputs.size(0), -1)
        cat = torch.cat((decoder_outputs, encoder_outputs), 1)
        W_a_cat = torch.tanh(self.W_a(cat))
        score = self.v_a(W_a_cat)
        a_t = nn.functional.softmax(score, dim=0)
        c_t = torch.mm(a_t.T, encoder_outputs)
        cat = torch.cat((c_t, decoder_output), 1)
        W_c_cat = self.W_c(cat)
        h_t_tilda = torch.tanh(W_c_cat)

        return h_t_tilda, a_t
