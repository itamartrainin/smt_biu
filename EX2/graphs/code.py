# Loss Plot
plt.plot(range(len(train_loss_scores)), train_loss_scores, label='Train')
plt.plot(range(len(dev_loss_scores)), dev_loss_scores, label='Dev')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('No Attention Loss')
plt.legend()
plt.show()

# SacreBleu Plot
plt.plot(range(len(train_bleu_scores)), train_bleu_scores, label='Train')
plt.plot(range(len(dev_bleu_scores)), dev_bleu_scores, label='Dev')
plt.xlabel('epoch')
plt.ylabel('bleu')
plt.ylim(0, 100)
plt.title('No Attention SacreBleu')
plt.legend()
plt.show()