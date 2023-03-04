'''
About Whatsapp Chat Analyzer Project

Today one of the trendy social media platforms one and only Whatsapp. It is one of the favorite social media platforms among all of us because of its attractive features. It has more than 2B users worldwide and “According to one survey an average user spends more than 195 minutes per week on WhatsApp”. How terrible the above statement is. Leave all these things and let’s understand what actually WhatsApp analyzer means?

WhatsApp Analyzer means we are analyzing our WhatsApp group activities. It tracks our conversation and analyses how much time we are spending on WhatsApp. Here I used different python libraries which help me to extract useful information from raw data. Here I choose my official WhatsApp group to analyze the pattern.

Data set is very dynamic. For seeing the analysis you can export your single user chat or group level with time stamp of 24 hrs format. And then upload the Chat export in text format

Used Streamlit to create web UI.Made all the required analytics at single user and group level with the supporting Visualization.Also, used NLP for sentiment analysis of your chat.

'''
#Import Libraries

import streamlit as st
import preprocessor,helper
from urlextract import URLExtract
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

extract = URLExtract()

st.sidebar.title('WHATSAPP CHAT ANALYZER')

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)



    ## fetch unique user
    user_list=df['user'].unique().tolist()
    #user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox('Show analysis', user_list)

    if st.sidebar.button("Show Analysis"):

        ## Total Message Count

        if selected_user == 'Overall':
            num_messages = df.shape[0]

            ##Total Words

            words = []
            for message in df['message']:
                words.extend(message.split())

            ## Total Media

            num_media = df[df['message'] == '<Media omitted>\n'].shape[0]

            ## Total Links

            links = []
            for message in df['message']:
                links.extend(extract.find_urls(message))

        else:

            ## Total Messages

            new_df = df[df['user'] == selected_user]
            num_messages = new_df.shape[0]

            ## Total Words

            words = []
            for message in new_df['message']:
                words.extend(message.split())

            ## Total Media

            num_media = new_df[new_df['message'] == '<Media omitted>\n'].shape[0]

            ## Total Links

            links = []
            for message in new_df['message']:
                links.extend(extract.find_urls(message))

        st.title('CHAT STATISTICS')
        col1, col2, col3, col4 = st.columns(4)

        #Chat Statistics Parameters

        with col1:
            st.header('Total Messages')
            st.title(num_messages)

        with col2:
            st.header('Total Words')
            st.title(len(words))

        with col3:
            st.header('Total Media')
            st.title(num_media)

        with col4:
            st.header('Total Links')
            st.title(len(links))


        #Monthly Timeline of Users
        st.title('Monthly Timeline')
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

        time = []
        for i in range(timeline.shape[0]):
            time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

        timeline['time'] = time

        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Daily Timeline of Users

        st.title('Daily Timeline')
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        daily_timeline = df.groupby('only_date').count()['message'].reset_index()
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Week Activity map
        st.title("ACTIVITY MAP")
        col1,col2 = st.columns(2)
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        w =  df['day_name'].value_counts()

        #Month Activity Map

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        m = df['month'].value_counts()

        with col1:
            st.header("Most Busy Day")
            fig,ax = plt.subplots()
            ax.bar(w.index,w.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            fig, ax = plt.subplots()
            ax.bar(m.index, m.values,color='yellow')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        #Activity Heat map
        st.title("ACTIVITY HEAT MAP")
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
        fig,ax = plt.subplots()
        ax=sns.heatmap(user_heatmap,annot=True)
        st.pyplot(fig)



        # finding the busiest user in group (Group Level)

        if selected_user == 'Overall':

            st.title('MOST BUSY USER')
            x = df['user'].value_counts().head()
            fig , ax = plt.subplots()
            col1 , col2 = st.columns(2)

            # Percentage of User
            df_new = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
                columns={'index': 'name', 'user': 'percent'})

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(df_new)

        # Wordcloud
        st.title('WORDCLOUD')
        f = open('stop_hinglish.txt', 'r')
        stop_words = f.read()

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        temp = df[df['user'] != 'group_notification']
        temp = temp[temp['message'] != '<Media omitted>\n']

       
        wc = WordCloud(width=300, height=300, min_font_size=10, background_color='black')

        df_wc= wc.generate(temp['message'].str.cat(sep=' '))
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        if selected_user != 'Overall':

            df = df[df['user'] == selected_user]

        temp = df[df['user']!='group_notification']
        temp = temp[temp['message']!='<Media omitted>\n']

        words = []

        for message in temp['message']:
            for word in message.lower().split():
                if word not in stop_words:
                    words.append(word)

        most_common_df = pd.DataFrame(Counter(words).most_common(20))
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('MOST COMMON WORDS')
        st.pyplot(fig)

        # Emoji Analysis
        st.title('EMOJI ANALYSIS')
        if selected_user != 'Overall':

            df = df[df['user'] == selected_user]

        y = []
        for i in df['message']:
            y.extend(emoji.emoji_lis(i))
        
        d = []
        for i in range(len(y)):
            d.append(y[i]['emoji'])
        c = pd.DataFrame(d)
        h = c.rename({0: 'emoji'}, axis=1).value_counts().to_frame().rename({0: 'count'}, axis=1).reset_index()


        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(h)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(h['count'],labels=h['emoji'],autopct="%0.2f")
            st.pyplot(fig)

        #Sentiment Analysis

        st.title('SENTIMENT ANALYSIS')
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        sentiments = SentimentIntensityAnalyzer()
        data = df.dropna()
        data["positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]]
        data["negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]]
        data["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]]

        z = data[['positive', 'negative', 'neutral']].sum().reset_index().rename(
            {'index': 'Sentiment_Type', 0: 'Count'}, axis=1)

        fig, ax = plt.subplots()
        ax.bar(z['Sentiment_Type'], z['Count'], color=['green', 'Red', 'Yellow'])
        plt.xlabel('Sentiment Type')
        plt.ylabel('Count')
        st.pyplot(fig)



'''
Result

Made Streamlit application to perform significant Monthly & Daily Timeline analysis, Emoji & Sentimental analysis, Time period activity and Top message analysis at Single user and group level user
Whatsapp chats using Natural Language Processing.

'''









