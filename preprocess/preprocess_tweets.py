import numpy as np
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

data_frame = pd.read_csv(r'..\data\tweets.csv')

# Graph 1: account tweets retweeted by hillary

# hillary_clinton_data = data_frame[(data_frame['handle'] == 'HillaryClinton') & data_frame['is_retweet'] == True][['original_author']]
#
# h_retweet_account_perc = hillary_clinton_data.groupby(['original_author']).size().reset_index(name='counts').sort_values('counts', ascending=False)
#
# h_labels = dict(zip(h_retweet_account_perc.original_author, h_retweet_account_perc.counts))
# wordcloud_retweet_h = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8).\
#     generate_from_frequencies(h_labels)
# plt.figure(figsize=(10, 10), facecolor=None)
# plt.imshow(wordcloud_retweet_h)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig('account_tweets_retweeted_by_hillary.png', bbox_inches='tight')
# # plt.show()
#
#
# # Graph 2: account tweets retweeted by trump
#
# trump_data = data_frame[(data_frame['handle'] == 'realDonaldTrump') &
#                         data_frame['is_retweet'] == True][['original_author']]
#
# t_retweet_account_perc = trump_data.groupby(['original_author']).size().reset_index(name='counts') \
#     .sort_values('counts', ascending=False)
#
# t_labels = dict(zip(t_retweet_account_perc.original_author, t_retweet_account_perc.counts))
# wordcloud_retweet_t = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8). \
#     generate_from_frequencies(t_labels)
# plt.figure(figsize=(10, 10), facecolor=None)
# plt.imshow(wordcloud_retweet_t)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig('account_tweets_retweeted_by_trump.png', bbox_inches='tight')
# # plt.show()
#
#
# # Graph 3: top n words by Hillary
#

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word.lower(), sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if word.lower() not in STOPWORDS and
                  len(word) > 3]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')
    # ReviewText = ReviewText.str.replace('co', ' ')
    ReviewText = ReviewText.str.replace('https', ' ')
    # ReviewText = ReviewText.str.replace('co', ' ')
    ReviewText = ReviewText.str.replace('@', ' ')

    return ReviewText
#
#
# hillary_tweets = data_frame[(data_frame['handle'] == 'HillaryClinton')][['text']]
# hillary_tweets['text'] = preprocess(hillary_tweets['text'])
# hillary_tweets['text'] = hillary_tweets['text'].map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))
#
#
# common_words = get_top_n_words(hillary_tweets['text'], 20)
#
# df1 = pd.DataFrame(common_words, columns=['words', 'count'])
# df1.plot(kind='bar', x='words', rot=85)
# plt.tight_layout()
# plt.savefig('top_n_words_by_Hillary.png', bbox_inches='tight')
# # plt.show()
#
#
# # Graph 4: top n words by trump
#
# trump_tweets = data_frame[(data_frame['handle'] == 'realDonaldTrump')][['text']]
# trump_tweets['text'] = preprocess(trump_tweets['text'])
# trump_tweets['text'] = trump_tweets['text'].map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))
#
#
# common_words = get_top_n_words(trump_tweets['text'], 20)
#
# df1 = pd.DataFrame(common_words, columns=['words', 'count'])
# df1.plot(kind='bar', x='words', rot=85)
# plt.tight_layout()
# plt.savefig('top_n_words_by_Trump.png', bbox_inches='tight')

# plt.show()

# Graph 5: wordcloud for VP
# vp_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\VP.csv', encoding='cp1252')
# vp_tweets = vp_dataframe['text']
#
# vp_tweets = preprocess(vp_tweets)
# vp_tweets = vp_tweets.map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))
#
#
# common_words = get_top_n_words(vp_tweets, 20)
#
#
# wordcloud_vp_t = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8). \
#     generate_from_frequencies(dict(common_words))
# plt.figure(figsize=(10, 10), facecolor=None)
# plt.imshow(wordcloud_vp_t)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig('vp_wc.png', bbox_inches='tight')

# Graph 6: word cloud for kamala

# kamala_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\kamala.csv', encoding='cp1252')
# kamala_dataframe = kamala_dataframe.dropna()
# kamala_tweets = kamala_dataframe['text']
#
# kamala_tweets = preprocess(kamala_tweets)
# kamala_tweets = kamala_tweets.map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))
#
#
# common_words = get_top_n_words(kamala_tweets, 20)
#
#
# wordcloud_kamala_t = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8). \
#     generate_from_frequencies(dict(common_words))
# plt.figure(figsize=(10, 10), facecolor=None)
# plt.imshow(wordcloud_kamala_t)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig('kamala_wc.png', bbox_inches='tight')

# Graph 7: word cloud for biden

# biden_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\biden.csv', encoding='cp1252')
# biden_dataframe = biden_dataframe.dropna()
# biden_tweets = biden_dataframe['text']
#
# biden_tweets = preprocess(biden_tweets)
# biden_tweets = biden_tweets.map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))
#
#
# common_words = get_top_n_words(biden_tweets, 20)
#
#
# wordcloud_biden_t = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8). \
#     generate_from_frequencies(dict(common_words))
# plt.figure(figsize=(10, 10), facecolor=None)
# plt.imshow(wordcloud_biden_t)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig('biden_wc.png', bbox_inches='tight')

# Graph 8: word cloud for potus

# potus_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\potus.csv', encoding='cp1252')
# potus_dataframe = potus_dataframe.dropna()
# potus_tweets = potus_dataframe['text']
#
# potus_tweets = preprocess(potus_tweets)
# potus_tweets = potus_tweets.map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))
#
#
# common_words = get_top_n_words(potus_tweets, 20)
#
#
# wordcloud_potus_t = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8). \
#     generate_from_frequencies(dict(common_words))
# plt.figure(figsize=(10, 10), facecolor=None)
# plt.imshow(wordcloud_potus_t)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig('potus_wc.png', bbox_inches='tight')

# Graph 9: word cloud for vp_kamala_mix

# vp_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\VP.csv', encoding='cp1252')
# kamala_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\kamala.csv', encoding='cp1252')
# kamala_dataframe = kamala_dataframe.dropna()
# vp_dataframe = vp_dataframe.dropna()
#
# vp_tweets = vp_dataframe['text']
# kamala_tweets = kamala_dataframe['text']
#
# vp_kamala_tweets = vp_tweets.append(kamala_tweets)
#
#
# vp_kamala_tweets = preprocess(vp_kamala_tweets)
# vp_kamala_tweets = vp_kamala_tweets.map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))
#
#
# common_words = get_top_n_words(vp_kamala_tweets, 20)
#
#
# wordcloud_vp_t = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8). \
#     generate_from_frequencies(dict(common_words))
# plt.figure(figsize=(10, 10), facecolor=None)
# plt.imshow(wordcloud_vp_t)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig('vp_kamala_wc.png', bbox_inches='tight')

# graph 10: word cloud for biden_potus_mix

biden_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\biden.csv', encoding='cp1252')
potus_dataframe = pd.read_csv(r'C:\Users\darsh\PycharmProjects\Twitter-Data-Analysis\data\potus.csv', encoding='cp1252')
biden_dataframe = biden_dataframe.dropna()
potus_dataframe = potus_dataframe.dropna()

potus_dataframe = potus_dataframe['text']
biden_dataframe = biden_dataframe['text']

biden_potus_tweets = potus_dataframe.append(biden_dataframe)


biden_potus_tweets = preprocess(biden_potus_tweets)
biden_potus_tweets = biden_potus_tweets.map(lambda text: " ".join(i.lower() for i in text.split() if i.lower() not in STOPWORDS))


common_words = get_top_n_words(biden_potus_tweets, 20)


wordcloud_vp_t = WordCloud(width=1000, height=1000, background_color='white', min_font_size=8). \
    generate_from_frequencies(dict(common_words))
plt.figure(figsize=(10, 10), facecolor=None)
plt.imshow(wordcloud_vp_t)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('biden_potus_wc.png', bbox_inches='tight')
