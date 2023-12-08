import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(df):
    '''Plot the data to see the distribution of the categories'''
    cnt_pro = df['Category'].value_counts()
    plt.figure(figsize=(12, 4))
    sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


def plot_history(history):
    '''Plot the history of the model'''
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model_accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model_loss.png')
