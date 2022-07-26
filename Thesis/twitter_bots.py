# Main Tokens
# consumer_key = '7cXG565ehcH9TO4XtgtjWiAF2'
# consumer_secret = 'J9qWmKfdiuTJxKwGvwDLASqSCMEwE1O85ta0QP8ARScnMAiTrR'
# access_token = '1351603554499387394-g2gyZg8hryr3QiqmRD1lnj5i16wAVV'
# access_token_secret = 'rZNyp3xngCUBK9rHlHmnD7Gu4OXRWRBUSHDPasBho8icL'
# bearer_token = 'AAAAAAAAAAAAAAAAAAAAAELZawEAAAAA6ItGxjfWQfyI3gSlZJtgfmBXQAo%3DkqWpeCvfK7q3OU6w6Rd0zEGpnKrDLUqS2dxJu2d2NjmMkifnWs'
# client_id = 'WlFtUXZhQnNrdkdDdzZVNDB2OFk6MTpjaQ'
# client_secret = 'arVmpRlANwEZUST55FcJsD8ZO0OhEOaM1Ng1VAku7-AgNcsQFY'

# email: rochelle_tunti@outlook.com
# password: S7ixKqK5Eg
# fullname: Rochelle Tunti
# username: RochelleTunti
# dob: 8/8/98

# import torch_mlir
import re
from datetime import time

import numpy as np
import requests
import torch
import tweepy
import json
from tweepy import OAuthHandler
from tweepy.streaming import Stream
import urllib.request, urllib.parse, urllib.error
import ssl
from deep_translator import GoogleTranslator
from searchtweets import ResultStream, load_credentials #, gen_request_parameters,
from nltk.tokenize import RegexpTokenizer
import hashlib

import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import matplotlib.pyplot as plt

import csv
import os

from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, pipeline
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import torch.nn.functional as F
import numpy as np
import torch
import time
from fastprogress.fastprogress import format_time, master_bar, progress_bar
from collections import defaultdict
import argparse

from AspectDetecter.Out_of_domain_ABSA.scripts.model import OodModel
from AspectDetecter.Out_of_domain_ABSA.scripts.data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from AspectDetecter.Out_of_domain_ABSA.SpanEmo.scripts.model import SpanEmo
from AspectDetecter.Out_of_domain_ABSA.SpanEmo.scripts.data_loader import DataClass as asp_DataClass



class MyListener(Stream):

    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, api, max_tweets, json_tweets_file):
        super(tweepy.Stream, self).__init__()
        self.num_tweets = 0
        self.max_tweets = max_tweets
        self.api = api
        self.json_tweets_file = json_tweets_file
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret


    def on_data(self, data):
        with open(self.json_tweets_file, 'a') as f:
            twitter_text = None
            if json.loads(data)['user']['location'] != None:
                try:
                    if json.loads(data)['user']['location'] != None:

                        twitter_text = json.loads(data)['text']
                    else:
                        twitter_text = json.loads(data)['retweeted_status']['extended_tweet']['full_text']

                    # f.write(data)  # This will store the whole JSON data in the file, you can perform some JSON filters
                    # twitter_text = json.loads(data)['retweeted_status']['extended_tweet']['full_text']
                    f.write(twitter_text + "\n")

                except BaseException as e:
                    # print("Error on_data: %s" % str(e))
                    twitter_text = json.loads(data)['text']
                    f.write(twitter_text + "\n")

                    return


                self.num_tweets += 1
                print(twitter_text.replace('\n', ' ').replace('\r', ' '))
                if self.num_tweets >= self.max_tweets:
                    raise Exception("Limit Reached")
                        # return

    def on_error(self, status):
        print('Error :', status)
        return False

# def user_timeline(user_name, user_id):
class AspectPredict(object):
    """
    Class to encapsulate evaluation on the test set. Based off the "Tonks Library"
    :param model: PyTorch model to use with the Learner
    :param test_data_loader: dataloader for all of the validation data
    :param model_path: path of the trained model
    """

    def __init__(self, model, test_data_loader, model_path):
        self.model = model
        self.test_data_loader = test_data_loader
        self.model_path = model_path

    def predict(self, device='cpu:0', pbar=None):
        """
        Evaluate the model on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: None
        """
        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.test_data_loader.dataset)
        preds_dict = {
            'y_true': np.zeros([current_size, 6]),
            'y_pred': np.zeros([current_size, 6])
        }
        start_time = time.time()
        with torch.no_grad():
            index_dict = 0
            with open('/Users/jordanharris/SCAPT-ABSA/AspectDetecter/Out_of_domain_ABSA/SpanEmo/predict_1.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['ID','text','food','restaurant','atmosphere','drinks','location','service'])
                for step, batch in enumerate(
                        progress_bar(self.test_data_loader, parent=pbar, leave=(pbar is not None))):
                    # inputs, targets, lengths, label_idxs, ids = batch

                    _, num_rows, y_pred, targets, ids = self.model(batch, device)
                    current_index = index_dict
                    preds_dict['y_true'][current_index: current_index + num_rows, :] = targets
                    preds_dict['y_pred'][current_index: current_index + num_rows, :] = y_pred
                    index_dict += num_rows
                    for i in range(num_rows):
                        writer.writerow([ids[i], "", ] + y_pred[i].tolist())

        y_true, y_pred = preds_dict['y_true'], preds_dict['y_pred']
        str_stats = []
        stats = [f1_score(y_true, y_pred, average="macro"),
                 f1_score(y_true, y_pred, average="micro"),
                 jaccard_score(y_true, y_pred, average="samples"),
                 precision_score(y_true, y_pred, average='micro'),
                 recall_score(y_true, y_pred, average='micro')
                 ]

        for stat in stats:
            str_stats.append(
                'NA' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.4f}'
            )
        str_stats.append(format_time(time.time() - start_time))
        headers = ['F1-Macro', 'F1-Micro', 'JS', 'precision_score', 'recall_score', 'Time']
        print(' '.join('{}: {}'.format(*k) for k in zip(headers, str_stats)))



class PredictSentiment:
    def __init__(self, model, fake_test_data_loader, real_test_data_loader, model_path):
        self.model = model
        self.fake_test_data_loader = fake_test_data_loader
        self.real_test_data_loader = real_test_data_loader
        self.model_path = model_path

    def predict(self, device='cpu:0', pbar=None):
        os.chdir('/Users/jordanharris/SCAPT-ABSA/AspectDetecter/Out_of_domain_ABSA/')
        # curr_state_dict = torch.load(self.model_path)

        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.fake_test_data_loader.dataset)
        preds_dict = {
            'y_pred': np.zeros([current_size, 1]),
            'y_true': np.zeros([current_size, 1])
        }
        start_time = time.time()
        # with torch.no_grad():
        #     index_dict = 0
        #     for step, batch in enumerate(progress_bar(self.fake_test_data_loader,
        #                                               parent=pbar,
        #                                               leave=(pbar is not None))):
        #         _, num_rows, y_pred, targets = self.model(batch, device)
        #         test = batch[0].data[0]
        #         aTK = self.fake_test_data_loader.sampler.data_source.tokenizer
        #
        #         current_index = index_dict
        #         sent = aTK.decode(test.tolist())
        #         y_pred = np.reshape(y_pred, (num_rows, 1))
        #         preds_dict['y_pred'][current_index:current_index + num_rows, :] = y_pred
        #         index_dict += num_rows

        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(progress_bar(self.real_test_data_loader,
                                                      parent=pbar,
                                                      leave=(pbar is not None))):
                _, num_rows, y_pred, targets = self.model(batch, device)

                test = batch[0].data[0]
                aTK = self.real_test_data_loader.sampler.data_source.tokenizer
                decoded = aTK.decode(test)

                tkn_twt = aTK.tokenize(
                    "The design is sleek and elegant, yet the case can stand up to a good beating.")

                inputs = aTK("The design is sleek and elegant, yet the case can stand up to a good beating.",
                             max_length=160,
                             padding='max_length',
                             truncation=True)

                self.model.eval()
                outputs, targets = [], []
                outs = []
                # need to push the data to device
                fly_batch = {}
                fly_batch[0] = torch.FloatTensor(inputs.data['input_ids'])
                fly_batch[1] = torch.FloatTensor(inputs.data['input_ids'])

                _, num_rows, y_pred, targets = self.model(input_ids=torch.FloatTensor(fly_batch))
                # _, num_rows, y_pred, targets = self.model.bert.bert.base_model(input_ids=inputs.data['input_ids'])
                                # attention_mask=inputs.data['attention_mask'],
                                # max_length=160)  # num_beams=8, early_stopping=True)

                dec = [aTK.decode(ids, skip_special_tokens=True) for ids in outs]
                target = [aTK.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

                outputs.extend(dec)
                targets.extend(target)

                # ==============# ==============# ==============# ==============# ==============# ==============

                current_index = index_dict
                targets = np.reshape(targets, (num_rows, 1))

                preds_dict['y_true'][current_index:current_index + num_rows, :] = targets
                index_dict += num_rows

        # ID	sentence	target	aspect	polarity
        # 1004293:5	Avoid this place!	place	restaurant	negative



        y_pred, y_true = preds_dict['y_pred'], preds_dict['y_true']
        stats = [precision_score(y_true, y_pred, average='micro'),
                 recall_score(y_true, y_pred, average='micro'),
                 f1_score(y_true, y_pred, average='micro'),
                 f1_score(y_true, y_pred, average='macro')]
        str_stats = []
        for stat in stats:
            str_stats.append(
                'NA' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.4f}'
            )

        headers = ['Precision', 'Recall', 'F1-micro', 'F1-macro']
        print(" ".join('{}: {}'.format(*k) for k in zip(headers, str_stats)))




def get_client(CONSUMER_KEY,CONSUMER_SECRET,BEARER_TOKEN,ACCESS_TOKEN,ACCESS_TOKEN_SECRET):
    client = tweepy.Client(bearer_token=BEARER_TOKEN,
                           consumer_key=CONSUMER_KEY,
                           consumer_secret=CONSUMER_SECRET,
                           access_token=ACCESS_TOKEN,
                           access_token_secret=ACCESS_TOKEN_SECRET, wait_on_rate_limit=True)
    return client

def pagination(client, user_id):
    responses = tweepy.Paginator(client.get_users_tweets, user_id,
                                 exclude='replies,retweets',
                                 max_results=100,
                                 expansions='referenced_tweets.id',
                                 tweet_fields=['created_at', 'public_metrics', 'entities'])
    return responses

def get_original_tweets(client, user_id):
    tweet_list = []
    responses = pagination(client, user_id)
    for response in responses:
        if response.data ==None:
            continue
        else:
            for tweets in response.data:
                tweet_list.append([tweets.text,
                                tweets['public_metrics']['like_count'],
                                tweets['public_metrics']['retweet_count'],
                                tweets['created_at'].date()])

    return tweet_list


def _test_creds(auth, bot_name='RochelleTunti'):
    api = tweepy.API(auth, wait_on_rate_limit=True)
    print(api.verify_credentials())

    # response = api.update_status("Its been 47 years.....")
    # print(response)
    user = api.get_user(screen_name=bot_name)
    new_friend = api.get_user(screen_name="DvrkdvysD")
    china_user = api.get_user(screen_name='jasonzhao_3')

    # bot_timeline = user_timeline(user_name=user.screen_name, user_id=user.id_str)

    print('Auth Screen Name:', api.verify_credentials().screen_name)
    print('User Screen Name:', user.screen_name)
    print('User ID:', user.id)
    print("The location is : " + str(user.location))
    print("The description is : " + user.description)
    print('User Follower Count:', user.followers_count)
    for friend in user.friends():
        print('Friend:', friend.screen_name)

def search_tweets(keywords, type='recent'):
    if type == 'recent':
        raw_tweets = client.search_recent_tweets(query=keywords,
                                            user_fields=['username', 'public_metrics', 'description', 'location', 'name', 'verified'],
                                            tweet_fields=['author_id', 'context_annotations', 'conversation_id', 'created_at', 'entities', 'geo', 'id',
                                                          'in_reply_to_user_id', 'lang', 'possibly_sensitive', 'public_metrics', 'referenced_tweets', 'reply_settings', 'source', 'text'],
                                            place_fields=['contained_within', 'country', 'country_code', 'geo', 'id', 'name', 'place_type'],
                                            expansions=['entities.mentions.username', 'geo.place_id', 'in_reply_to_user_id', 'referenced_tweets.id', 'referenced_tweets.id.author_id'],
                                            start_time='2022-06-22T21:25:00Z',
                                            end_time='2022-06-29T00:00:00Z',
                                            # geo_code=geoc,
                                            max_results=100)
    else:
        raw_tweets = client.search_all_tweets(query=keywords,
                                          user_fields=['username', 'public_metrics', 'description', 'location'],
                                          tweet_fields=['created_at', 'geo', 'public_metrics', 'text', 'attachments',
                                                        'referenced_tweets',
                                                        'entities'],
                                          # place_fields=['place_type', 'geo'],
                                          expansions='geo.place_id', max_results=100)
    collect = []
    tweets = json.loads(raw_tweets.content)
    # with open('usa_tweets.txt', 'a') as the_file:
        # the_file.write('Hello\n')
    for idx, t in enumerate(tweets['data']):
        try:
            if t['referenced_tweets'][0]['id'] == tweets['includes']['tweets'][idx]['id']:
                full_rt_text = tweets['includes']['tweets'][idx]['text']
                clean = full_rt_text.strip('\n')
        except:
            clean = t['text'].strip('\n')

        # clean = full_rt_text.strip('\n')
        # clean = t['text'].strip('\n')
        translated = GoogleTranslator(source='auto', target='en').translate(clean)
        collect.append(translated)
        try:
            print(api.get_user(user_id=t['author_id']).location, ':////////:', translated)
            print('------------------------------------------------')
        except BaseException as e:
            print('tweet deleted')
        print('_______ End _______')
        return collect

def search_cursor(keywords):
    collect =[ ]
    for x in tweepy.Cursor(api.search_tweets,
                           q=keywords,
                           tweet_mode="extended",
                           # geocode="39.925533, 32.866287, 30mi",
                           since='2022-06-07T00:25:00Z',
                           fromDate='2022-06-17T00:25:00Z',
                           toDate='2022-06-24T00:00:00Z',
                           lang="en",
                           result_type="mixed").items(200):
        clean = x.full_text.strip('\n')

        translated = GoogleTranslator(source='auto', target='en').translate(clean)

        collect.append(x.full_text)
    return collect

def stream_tweets(keywords, geo=str):
    # json_tweets_file = '_'.join(keywords)+'.jsonl'
    json_tweets_file = 'china_test.jsonl'
    # # You can increase this value to retrieve more tweets but remember the rate limiting
    max_tweets = 200
    twitter_stream = MyListener(consumer_key, consumer_secret, access_token, access_token_secret, api, max_tweets, json_tweets_file)
    # # # Add your keywords and other filters
    twitter_stream.filter(track=keywords, locations=geo, languages=["en"])
    print('_______ End _______')

def raw_to_df(raw_data):
    with open(raw_data) as f:
        lines = f.read().splitlines()

    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    print(len(df), 'tweets')
    df.head()

def cleaner(input_path, output_path):
    # __________________________Clean Results______________________________#

    # with open('reddit_data', 'a') as f:  # You can also print your tweets here

    cleaned_tweets = []
    regex_pattern = re.compile(pattern="["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
    #
    pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
    re_list = ['@[A-Za-z0–9_]+', '#']
    combined_re = re.compile( '|'.join( re_list) )
    # for clean_prep in twitter_stream.listener.extended_text:
    with open('./data/coronavirus_covid_omicorn_ukraine_russia_poland.jsonl') as f:
        line_prep = f.read().splitlines()
    clean_prep = []
    for line in line_prep:
        try:
            if json.loads(line)['is_quote_status'] ==True:
                clean_prep.append(json.loads(line)['quoted_status']['extended_tweet']['full_text'])
            else:
                clean_prep.append(json.loads(line)['retweeted_status']['extended_tweet']['full_text'])
        except:
            try:
                clean_prep.append(json.loads(line)['text'])
            except:
                continue
    with open('clean__' + output_path, 'a') as f:
        for cp in clean_prep:
            clean = re.sub(regex_pattern, '', cp)
            # replaces pattern with ''
            clean_tweets_1 = re.sub(pattern, '', clean)
            clean_tweets_2 = re.sub(combined_re, '', clean_tweets_1)
            clean_tweets_3 = re.sub('\n', '', clean_tweets_2)
            clean_tweets_4 = re.sub('\'', '', clean_tweets_3)
            cleaned_tweets.append(clean_tweets_4)
            f.write(str(clean_tweets_4) + '\n')

    clean_strings_df = pd.Series(cleaned_tweets).str.cat(sep=' ')
    return clean_strings_df

def delete_dupes(output_file_path, input_file_path):
    # 1
    # output_file_path = "C:/out.txt"
    # input_file_path = "C:/in.txt"
    # 'usa_tweets.txt'
    # 2
    openFile = open(input_file_path, "r")
    writeFile = open(output_file_path, "w")
    # Store traversed lines
    tmp = set()
    for txtLine in openFile:
        # Check new line
        if txtLine not in tmp:
            writeFile.write(txtLine)
            writeFile.write('------------------------------------------------\n')
            # Add new traversed line to tmp
            tmp.add(txtLine)
    openFile.close()
    writeFile.close()


if __name__ == '__main__':

    # delete_dupes('usa_tweets_clean.txt', 'usa_tweets.txt')

    # # ________________________Authorize Twitter API________________________________#
    # consumer_key = '7cXG565ehcH9TO4XtgtjWiAF2'
    # consumer_secret = 'J9qWmKfdiuTJxKwGvwDLASqSCMEwE1O85ta0QP8ARScnMAiTrR'
    # access_token = '1351603554499387394-g2gyZg8hryr3QiqmRD1lnj5i16wAVV'
    # access_token_secret = 'rZNyp3xngCUBK9rHlHmnD7Gu4OXRWRBUSHDPasBho8icL'
    # bearer_token = 'AAAAAAAAAAAAAAAAAAAAAELZawEAAAAA6ItGxjfWQfyI3gSlZJtgfmBXQAo%3DkqWpeCvfK7q3OU6w6Rd0zEGpnKrDLUqS2dxJu2d2NjmMkifnWs'
    # client_id = 'WlFtUXZhQnNrdkdDdzZVNDB2OFk6MTpjaQ'
    # client_secret = 'arVmpRlANwEZUST55FcJsD8ZO0OhEOaM1Ng1VAku7-AgNcsQFY'
    #
    # auth = OAuthHandler(consumer_key, consumer_secret, callback="oob")
    # # print(auth.get_authorization_url())
    # # Enter that PIN to continue
    # # verifier = input("PIN (oauth_verifier= parameter): ")
    # # Complete authenthication
    # # bot_token, bot_secret = auth.get_access_token(verifier)
    # # auth.set_access_token(bot_token, bot_secret)
    #
    #
    # auth.set_access_token(access_token, access_token_secret)
    # api = tweepy.API(auth, wait_on_rate_limit=True)
    # bot_name = 'RochelleTunti'
    # _test_creds(auth, bot_name=bot_name)
    #
    #
    # city = 'USA'
    # places = api.search_geo(query='{}'.format(city), granularity="country")
    # place_id = places[0].id
    # # place = api.geo_id(place_id=place_id)
    # # locations = api.reverse_geocode(39.925533,  32.866287)
    # # place_id = locations[0].id
    #
    # client = tweepy.Client(bearer_token=bearer_token,
    #                        consumer_key=consumer_key,
    #                        consumer_secret=consumer_secret,
    #                        access_token=access_token,
    #                        access_token_secret=access_token_secret,
    #                        return_type=requests.Response,
    #                        wait_on_rate_limit=True)
    # # client.get_user(username=user.screen_name)
    # # client.get_users_tweets(id=user.id)
    #
    # keywords_cn = '("Xinjiang" OR "Uyghurs" OR "Muslim" OR "muslim" OR "China" OR "china" OR "Peoples War on Terror" OR "Peoples War") ("Ethnic" OR "Ethnicity" OR "Minorities" OR "Uyghurs" OR "China" OR "CCP" OR "Taiwan" OR "Tibet" OR "Uzbek" OR "Kazak" OR "Muslim" OR "Pompeo" OR "PRC" OR "Kazakhstan" OR "Chinese" OR "Abuse" OR "SA" OR "R@pe" OR "Russia" OR "Ukraine" OR "Putin" OR "Xi Jinping") -is:retweet'
    # keywords_cn = '("Xinjiang" OR "Uyghurs" OR "Muslim" OR "muslim" OR "counter terrorism" OR "peoples war" OR "Peoples War on Terror" OR "People’s War") ("Ethnic" OR "Ethnicity" OR "Minorities" OR "Uyghurs" OR "China" OR "CCP" OR "Taiwan" OR "Tibet" OR "Uzbek" OR "Kazak" OR "Muslim" OR "PRC" OR "Kazakhstan" OR "Chinese" OR "Abuse" OR "SA" OR "R@pe" OR "Russia" OR "Ukraine" OR "Putin" OR "education") -is:retweet'
    # keywords_ru = '("Kiev" OR "Ukranians" OR "Ukraine" OR "refugees") ("Occupation" OR "денацификация" OR "Nazi" OR "Denazify" OR "War" OR "Racism" OR "Africans" OR "Flee" OR "Uzbek" OR "Kazak" OR "Muslim" OR "Minorities" OR "PRC" OR "Polish" OR "fight" OR "Abducted" OR "Killed" OR "Russia" OR "Ukraine" OR "Putin" OR "Xi Jinping" OR "Soviet")'
    #
    # keywords_us = '("Anti-Black" OR "Polish" OR "yt people" OR "palm colored") ("black people" OR "reparations" OR "poc" OR "african" OR "blacness" OR "blk")'
    # # keywords_us = '("black people" OR "reparations" OR "poc" OR "african" OR "blackness" OR "blk")'
    # # keywords_us = '("Anti-Black" OR "yt" OR "palm colored") ("black people" OR "reparations" OR "poc" OR "african" OR "blacness" OR "blk")'
    #
    # keywords_tt = '("Anti-Black" OR "deleted" OR "banned from tiktok" OR "shadowbanned" OR "taken down" OR "homophobic" OR "removed" OR "transphobic" OR "censorship" OR "slang" OR "coded" OR "codewords" OR "#XinjiangOnline") Tiktok OR "Amir Locke"'
    #
    # keywords = ["Xinjiang", 'Uyghurs', 'China', ' CCP regime', 'Uzbek', 'Kazak', 'Muslim', 'Pompeo', 'PRC', 'Kazakhstan']
    # # keywords = ["coronavirus", "covid", "omicorn", "ukraine", "russia", "poland", 'russian', 'ukranian', 'german', 'nazi'
    # #             "war", "weapons", "refugee", "detainee", "support", "refugees", "usa", "Volodymyr Zelenskyy",
    # #             "Vladimir Putin", "Joe Biden", "Joe Byron", "China", "Xi Jinping", 'putin',
    # #             "Andrzej Duda", "EU", "NATO", "Oil", "Gas", "Sanction", "Subvariant"]
    #
    # search_tweets = search_tweets(keywords_tt, type='recent')
    # # stream_tweets(keywords_ru)

    # ------------------------------------------------

    device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Train model on multiple cards')
    parser.add_argument('--model_path_sent', type=str)
    parser.add_argument('--real_test_path', type=str)
    parser.add_argument('--fake_test_path', type=str)

    parser.add_argument('--max_length', type=str)
    parser.add_argument('--bert_type', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--lang', type=str)

    parser.add_argument('--model_path_asp', type=str)
    parser.add_argument('--test_path', type=str)

    args = parser.parse_args()
    device = torch.device('cpu')
    # ==============Aspect Detector ==============# ==============# ==============# ==============# ==============

    test_dataset = asp_DataClass(args, vars(args)['test_path'])
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=int(vars(args)['test_batch_size']),
                                  shuffle=False)
    print('The number of Test batches: ', len(test_data_loader))

    asp_model = SpanEmo(lang=vars(args)['lang'], bert_type=vars(args)['bert_type'])
    learn = AspectPredict(asp_model, test_data_loader, model_path=vars(args)['model_path_asp'])
    learn.predict(device=device)

    # testing    #############################################################################
    from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sd_path = "/Users/jordanharris/SCAPT-ABSA/AspectDetecter/Out_of_domain_ABSA/SpanEmo/models/SemEval_2022-07-26-19:08:19_checkpoint.pt"
    model_dict = torch.load(sd_path)
    asp_model.load_state_dict(state_dict=model_dict)
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased",
                                                            state_dict=model_dict,
                                                            num_labels=6,
                                                            # num_labels=int(vars(args)['num_classes']),
                                                            # id2label={0 : "food", 1 : "restaurant", 2 : "atmosphere"3 :"drinks", 4 : "location", 5 : "service"},
                                                            # label2id={"food": 0, "restaurant": 1, "atmosphere": 2, "drinks": 3, "location": 4, "service": 5},
                                                            label2id={"None": 0, "Aspect": 1},
                                                            id2label={0: "None", 1: "Aspect"},
                                                            )

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, function_to_apply='softmax')
    tweet = ["The design is sleek and elegant, yet the case can stand up to a good beating."]
    prediction = pipe(tweet, return_all_scores=True)
    print()
    # testing    #############################################################################


    # ==============SentiMENTdeTECTOR ==============# ==============# ==============# ==============# ==============

    # real_test_dataset = DataClass(args, vars(args)['real_test_path'])
    # fake_test_dataset = DataClass(args, vars(args)['fake_test_path'])
    # real_test_data_loader = DataLoader(real_test_dataset,
    #                                    batch_size=int(vars(args)['test_batch_size']),
    #                                    shuffle=False)
    # fake_test_data_loader = DataLoader(fake_test_dataset,
    #                                    batch_size=int(vars(args)['test_batch_size']),
    #                                    shuffle=False)
    #
    # print('The number of Test batches: ', len(fake_test_data_loader))
    # #############################################################################
    # # Run the models on a Test set
    # #############################################################################
    # model = OodModel(model_type=vars(args)['bert_type'])
    # learn = PredictSentiment(model, fake_test_data_loader, real_test_data_loader,
    #                     model_path='models/' + vars(args)['model_path_sent'])
    # learn.predict(device=device)
    #
    # # ------------------------------------------------


#     #https://seinecle.github.io/gephi-tutorials/generated-html/twitter-streaming-importer-en.html
#     #https://seinecle.github.io/gephi-tutorials/generated-pdf/importing-csv-data-in-gephi-en.pdf
#     # https: // onnxruntime.ai / docs / build / inferencing  # cross-compiling-for-arm-with-simulation-linuxwindows
#     print('tweet search data')
#     searching = keywords_us + " place:" + place_id + " include:antisocial"
#     # searching = keywords_tt + " -filter:retweets include:antisocial"
#
#     # tweets = json.loads(raw_tweets.content)
#     with open('usa_tweets.txt', 'a') as the_file:
#         # line_prep = the_file.read().splitlines()
#
#
#         raw1 = search_tweets(keywords_us, type='all')
#         for t in raw1:
#             try:
#                 the_file.write(t + '\n')
#                 # the_file.write(api.get_user(user_id=t['author_id']).location, ':////////:', t)
#                 the_file.write('------------------------------------------------\n')
#             except BaseException as e:
#                 the_file.write('tweet deleted')
#
#         # raw2 = search_cursor(searching)
#         # for t in raw2:
#         #     try:
#         #         # the_file.write(t + '\n')
#         #         pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
#         #
#         #         clean = re.sub(pattern, '', t)
#         #         clean = re.sub('\n', ' ', t)
#         #
#         #         the_file.write(api.get_user(user_id=t['author_id']).location, ':////////:', clean)
#         #         the_file.write('------------------------------------------------\n')
#         #     except BaseException as e:
#         #         the_file.write('tweet deleted')
#         # the_file.write(str(x.user.location) + ':////////:' + translated)
#
#
#
#
# {"text": "The design is sleek and elegant, yet the case can stand up to a good beating.",
#  "tokens": ["the", "design", "is", "sleek", "and", "elegant", ",", "yet", "the", "case", "can", "stand", "up", "to", "a", "good", "beating", "."],
#  "aspect_terms": [{"aspect_term": "design", "left_index": 4, "right_index": 10, "sentiment": 5},
#                   {"aspect_term": "case", "left_index": 41, "right_index": 45, "sentiment": 5},
#                   {"aspect_term": "stand", "left_index": 50, "right_index": 55, "sentiment": 5}]}
# {"text": "The iBook comes with an awesome set of features--pretty much everything you might need is already part of the package, including FireWire, CD-RW drive, and 10/100 Ethernet.",
#  "tokens": ["the", "ibook", "comes", "with", "an", "awesome", "set", "of", "features", "--", "pretty", "much", "everything", "you", "might", "need", "is", "already", "part", "of", "the", "package", ",", "including", "firewire", ",", "cd-rw", "drive", ",", "and", "10/100", "ethernet", "."],
#  "aspect_terms": [{"aspect_term": "features", "left_index": 39, "right_index": 47, "sentiment": 5},
#                   {"aspect_term": "drive", "left_index": 145, "right_index": 150, "sentiment": 5}]}
# {"text": "Despite having a relatively small screen (12.",
#  "tokens": ["despite", "having", "a", "relatively", "small", "screen", "(", "12", "."],
#  "aspect_terms": [{"aspect_term": "screen", "left_index": 34, "right_index": 40, "sentiment": 5}]}
#

    # res = []
    # ner_pipe = pipeline('ner')
    # sequence = """Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO,
    # therefore very close to the Manhattan Bridge which is visible from the window."""
    # for entity in ner_pipe(sequence):
    #     res.append(entity)
    #     print(entity)
