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
import requests
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

    # ________________________Authorize Twitter API________________________________#
    consumer_key = '7cXG565ehcH9TO4XtgtjWiAF2'
    consumer_secret = 'J9qWmKfdiuTJxKwGvwDLASqSCMEwE1O85ta0QP8ARScnMAiTrR'
    access_token = '1351603554499387394-g2gyZg8hryr3QiqmRD1lnj5i16wAVV'
    access_token_secret = 'rZNyp3xngCUBK9rHlHmnD7Gu4OXRWRBUSHDPasBho8icL'
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAELZawEAAAAA6ItGxjfWQfyI3gSlZJtgfmBXQAo%3DkqWpeCvfK7q3OU6w6Rd0zEGpnKrDLUqS2dxJu2d2NjmMkifnWs'
    client_id = 'WlFtUXZhQnNrdkdDdzZVNDB2OFk6MTpjaQ'
    client_secret = 'arVmpRlANwEZUST55FcJsD8ZO0OhEOaM1Ng1VAku7-AgNcsQFY'

    auth = OAuthHandler(consumer_key, consumer_secret, callback="oob")
    # print(auth.get_authorization_url())
    # Enter that PIN to continue
    # verifier = input("PIN (oauth_verifier= parameter): ")
    # Complete authenthication
    # bot_token, bot_secret = auth.get_access_token(verifier)
    # auth.set_access_token(bot_token, bot_secret)


    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    bot_name = 'RochelleTunti'
    _test_creds(auth, bot_name=bot_name)


    city = 'USA'
    places = api.search_geo(query='{}'.format(city), granularity="country")
    place_id = places[0].id
    # place = api.geo_id(place_id=place_id)
    # locations = api.reverse_geocode(39.925533,  32.866287)
    # place_id = locations[0].id

    client = tweepy.Client(bearer_token=bearer_token,
                           consumer_key=consumer_key,
                           consumer_secret=consumer_secret,
                           access_token=access_token,
                           access_token_secret=access_token_secret,
                           return_type=requests.Response,
                           wait_on_rate_limit=True)
    # client.get_user(username=user.screen_name)
    # client.get_users_tweets(id=user.id)

    keywords_cn = '("Xinjiang" OR "Uyghurs" OR "Muslim" OR "muslim" OR "China" OR "china" OR "Peoples War on Terror" OR "Peoples War") ("Ethnic" OR "Ethnicity" OR "Minorities" OR "Uyghurs" OR "China" OR "CCP" OR "Taiwan" OR "Tibet" OR "Uzbek" OR "Kazak" OR "Muslim" OR "Pompeo" OR "PRC" OR "Kazakhstan" OR "Chinese" OR "Abuse" OR "SA" OR "R@pe" OR "Russia" OR "Ukraine" OR "Putin" OR "Xi Jinping") -is:retweet'
    keywords_cn = '("Xinjiang" OR "Uyghurs" OR "Muslim" OR "muslim" OR "counter terrorism" OR "peoples war" OR "Peoples War on Terror" OR "People’s War") ("Ethnic" OR "Ethnicity" OR "Minorities" OR "Uyghurs" OR "China" OR "CCP" OR "Taiwan" OR "Tibet" OR "Uzbek" OR "Kazak" OR "Muslim" OR "PRC" OR "Kazakhstan" OR "Chinese" OR "Abuse" OR "SA" OR "R@pe" OR "Russia" OR "Ukraine" OR "Putin" OR "education") -is:retweet'
    keywords_ru = '("Kiev" OR "Ukranians" OR "Ukraine" OR "refugees") ("Occupation" OR "денацификация" OR "Nazi" OR "Denazify" OR "War" OR "Racism" OR "Africans" OR "Flee" OR "Uzbek" OR "Kazak" OR "Muslim" OR "Minorities" OR "PRC" OR "Polish" OR "fight" OR "Abducted" OR "Killed" OR "Russia" OR "Ukraine" OR "Putin" OR "Xi Jinping" OR "Soviet")'

    keywords_us = '("Anti-Black" OR "Polish" OR "yt people" OR "palm colored") ("black people" OR "reparations" OR "poc" OR "african" OR "blacness" OR "blk")'
    # keywords_us = '("black people" OR "reparations" OR "poc" OR "african" OR "blackness" OR "blk")'
    # keywords_us = '("Anti-Black" OR "yt" OR "palm colored") ("black people" OR "reparations" OR "poc" OR "african" OR "blacness" OR "blk")'

    keywords_tt = '("Anti-Black" OR "deleted" OR "banned from tiktok" OR "shadowbanned" OR "taken down" OR "homophobic" OR "removed" OR "transphobic" OR "censorship" OR "slang" OR "coded" OR "codewords" OR "#XinjiangOnline") Tiktok OR "Amir Locke"'

    keywords = ["Xinjiang", 'Uyghurs', 'China', ' CCP regime', 'Uzbek', 'Kazak', 'Muslim', 'Pompeo', 'PRC', 'Kazakhstan']
    # keywords = ["coronavirus", "covid", "omicorn", "ukraine", "russia", "poland", 'russian', 'ukranian', 'german', 'nazi'
    #             "war", "weapons", "refugee", "detainee", "support", "refugees", "usa", "Volodymyr Zelenskyy",
    #             "Vladimir Putin", "Joe Biden", "Joe Byron", "China", "Xi Jinping", 'putin',
    #             "Andrzej Duda", "EU", "NATO", "Oil", "Gas", "Sanction", "Subvariant"]

    search_tweets = search_tweets(keywords_tt, type='recent')
    # stream_tweets(keywords_ru)



    #https://seinecle.github.io/gephi-tutorials/generated-html/twitter-streaming-importer-en.html
    #https://seinecle.github.io/gephi-tutorials/generated-pdf/importing-csv-data-in-gephi-en.pdf
    # https: // onnxruntime.ai / docs / build / inferencing  # cross-compiling-for-arm-with-simulation-linuxwindows
    print('tweet search data')
    searching = keywords_us + " place:" + place_id + " include:antisocial"
    # searching = keywords_tt + " -filter:retweets include:antisocial"

    # tweets = json.loads(raw_tweets.content)
    with open('usa_tweets.txt', 'a') as the_file:
        # line_prep = the_file.read().splitlines()


        raw1 = search_tweets(keywords_us, type='all')
        for t in raw1:
            try:
                the_file.write(t + '\n')
                # the_file.write(api.get_user(user_id=t['author_id']).location, ':////////:', t)
                the_file.write('------------------------------------------------\n')
            except BaseException as e:
                the_file.write('tweet deleted')

        # raw2 = search_cursor(searching)
        # for t in raw2:
        #     try:
        #         # the_file.write(t + '\n')
        #         pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
        #
        #         clean = re.sub(pattern, '', t)
        #         clean = re.sub('\n', ' ', t)
        #
        #         the_file.write(api.get_user(user_id=t['author_id']).location, ':////////:', clean)
        #         the_file.write('------------------------------------------------\n')
        #     except BaseException as e:
        #         the_file.write('tweet deleted')
        # the_file.write(str(x.user.location) + ':////////:' + translated)




{"text": "The design is sleek and elegant, yet the case can stand up to a good beating.",
 "tokens": ["the", "design", "is", "sleek", "and", "elegant", ",", "yet", "the", "case", "can", "stand", "up", "to", "a", "good", "beating", "."],
 "aspect_terms": [{"aspect_term": "design", "left_index": 4, "right_index": 10, "sentiment": 5},
                  {"aspect_term": "case", "left_index": 41, "right_index": 45, "sentiment": 5},
                  {"aspect_term": "stand", "left_index": 50, "right_index": 55, "sentiment": 5}]}
{"text": "The iBook comes with an awesome set of features--pretty much everything you might need is already part of the package, including FireWire, CD-RW drive, and 10/100 Ethernet.",
 "tokens": ["the", "ibook", "comes", "with", "an", "awesome", "set", "of", "features", "--", "pretty", "much", "everything", "you", "might", "need", "is", "already", "part", "of", "the", "package", ",", "including", "firewire", ",", "cd-rw", "drive", ",", "and", "10/100", "ethernet", "."],
 "aspect_terms": [{"aspect_term": "features", "left_index": 39, "right_index": 47, "sentiment": 5},
                  {"aspect_term": "drive", "left_index": 145, "right_index": 150, "sentiment": 5}]}
{"text": "Despite having a relatively small screen (12.",
 "tokens": ["despite", "having", "a", "relatively", "small", "screen", "(", "12", "."],
 "aspect_terms": [{"aspect_term": "screen", "left_index": 34, "right_index": 40, "sentiment": 5}]}

