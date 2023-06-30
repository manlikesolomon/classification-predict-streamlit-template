"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
import os
# Data dependencies
import pandas as pd
import string
import nltk
import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
import base64
st.set_option('deprecation.showPyplotGlobalUse', False)

# Vectorizer
news_vectorizer = open("resources/vectorizer_max.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def preprocess(message):
    punctuations = string.punctuation
    stopwords_english = stopwords.words('english')
    # remove hashtag(#) from tweet
    message = re.sub('#','',message)
    # remove oldstyle 'RT' from our tweets
    message = re.sub(r'^RT[\s]+','',message)
    # remove hyperlink
    message = re.sub(r'https?://[^s\n\r]+','',message)
    # Remove "��"
    message = re.sub("��", "",message)
    # Remove text starting with "@"
    message = re.sub(r"@\w+\s?", "", message)
    # Remove newline followed by any character/digit
    message = re.sub(r"\n.", "", message)
    # convert our text to lower case
    message = message.lower()
    # remove punctuations
    message = ''.join([word for word in message if word not in punctuations])
    # tokenize
    tokenizer = TreebankWordTokenizer()
    message = tokenizer.tokenize(message)
    # remove stop words
    message = [word for word in message if word not in stopwords_english]

    # lemmatize text
    lemmatizer = WordNetLemmatizer()
    lem_message = []

    for word in message:
        lem_message.append(lemmatizer.lemmatize(word))

    return " ".join(lem_message)

#Function to perform exploratory data analysis of uploaded csv
#function to provide word-lengths
def word_length(data):
	length = len(data)
	return length

#function to create a graph of word lengths
def length_graph(data):
	fig = plt.hist(data['message_length'])
	plt.title("Distribution Of Tweet Lenghts")
	plt.xlabel("Lengths")
	plt.ylabel('Frequency')
	st.pyplot()

#function to create dataframes for all classes 
def classes(data):
	pro = data[data['Sentiments'] == 'Pro']
	news = data[data['Sentiments'] == 'News']
	neutral = data[data['Sentiments'] == 'Neutral']
	anti = data[data['Sentiments'] == 'Anti']
	return pro, news, neutral, anti

#function to display graphs of word lengths of each class
def length_graph_class(data):
	pro = data[data['Sentiments'] == 'Pro']
	news = data[data['Sentiments'] == 'News']
	neutral = data[data['Sentiments'] == 'Neutral']
	anti = data[data['Sentiments'] == 'Anti']
	fig = plt.figure(figsize=(10,10))
	plt.subplot(2,2,1)
	plt.hist(pro['message_length'],bins=5)
	plt.title('Distribution of Pro Class Tweet Lengths')

	plt.subplot(2,2,2)
	plt.hist(news['message_length'],bins=5)
	plt.title('Distribution of News Class Tweet Lengths')

	plt.subplot(2,2,3)
	plt.hist(anti['message_length'],bins=5)
	plt.title('Distribution of Anti Class Tweet Lengths')

	plt.subplot(2,2,4)
	plt.hist(neutral['message_length'],bins=5)
	plt.title('Distribution of Neutral Class Tweet Lengths')

	plt.tight_layout()
	st.pyplot()

#function to create a dataframe containing average tweet lenghts for every class
def avg_lenght_df(data):
	pro_avg = round(data.loc[data['Sentiments'] == 'Pro', "message_length"].mean(),3)
	news_avg = round(data.loc[data['Sentiments'] == 'News', "message_length"].mean(),3)
	anti_avg = round(data.loc[data['Sentiments'] == 'Anti', "message_length"].mean(),3)
	neutral_avg = round(data.loc[data['Sentiments'] == 'Neutral', "message_length"].mean(),3)
	df = pd.DataFrame({
        'Class': ['Pro', 'News', 'Anti', 'Neutral'],
        'Average Length': [pro_avg, news_avg, anti_avg, neutral_avg]
    })
	return df

#function to create a graph of top ten words for each sentiment
def top_words(df, title_class, n=10):
    preprocessed_text = df['Clean_Tweets'].apply(lambda x: x.split())
    too_common = ['climate', 'change', 'global', 'warming']
    all_words = [word for sublist in preprocessed_text for word in sublist if word not in too_common]
    word_freq = Counter(all_words)
    most_common = word_freq.most_common(n)
    words, frequencies = zip(*most_common)

    # Display the bar chart using Streamlit
    st.subheader(f'Top {n} Most Frequent Words in {title_class}')
    fig, ax = plt.subplots()
    ax.bar(words, frequencies)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Top {n} Most Frequent Words in {title_class}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
#function to display distribution of sintiments in predictions
def dominate(data):
	plt.figure(figsize=(8, 6))
	sentiment_counts = data['Sentiments'].value_counts()
	plt.bar(sentiment_counts.index, sentiment_counts.values)
	plt.xlabel('Sentiments')
	plt.ylabel('Count')
	plt.title('Distribution of Sentiments')
	plt.xticks(rotation=45)
	st.pyplot()



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#sets the background image
	with open('design.css') as designs:
		st.markdown(f"<style>{designs.read()}</style>", unsafe_allow_html=True)

	logo = Image.open('Images/GS.png')
	#logo2 = Image.open('Images/gg.png')
	st.image(logo, use_column_width=True)
	#st.image(logo2)
	#st.markdown(f'<img src="http://localhost:8501/media/6b4b9df26b5cb297db50888a170735f06acdcd490592b420e511ae81.png" alt="logo" class="logo-image">', unsafe_allow_html=True)
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	logo1 = Image.open('Images/ggb2.png')
	st.sidebar.image(logo1, use_column_width=True)
	options = ["Home", "Information", "Explore", "Predict Single Input", "Predict CSV", 'Documentation']
	selection = st.sidebar.selectbox("Choose Option", options)
	
	#building out Homepage
	if selection == "Home":
		st.markdown('# HOME')
		if st.checkbox('About US'):
			st.info("## Welcome to GreenGraph Solutions!🌟")
			st.markdown('---')
			st.markdown('We are not just a data science group; we are visionaries, innovators, and pioneers in the world of data-driven solutions. With a relentless passion for harnessing the power of data, we have embarked on a mission to shape the future through our cutting-edge predictive analytics capabilities. \n- At GreenGraph Solutions, we believe that data holds the key to unlocking endless possibilities. We combine our expertise in machine learning, artificial intelligence, and advanced analytics to uncover hidden insights, solve complex problems, and drive informed decision-making. \n- What sets us apart is our unwavering commitment to delivering excellence. We leverage the latest technologies and employ state-of-the-art methodologies to ensure the highest level of accuracy, reliability, and scalability in our predictions. \n Beyond our technical prowess, we pride ourselves on our collaborative approach and client-centric mindset. Our dedication to building long-term relationships and providing exceptional customer service sets us apart as a trusted advisor and strategic partner. \n- Join us on this exhilarating journey as we redefine what is possible in the world of data science. Let GreenGraph Solutions be your trusted guide as we navigate the vast landscape of data, transforming it into actionable insights and unlocking the true potential of your business. Together, let us shape a brighter, data-driven future. \n- GreenGraph Solutions - **Where Data Meets Possibilities🌈.**')
		

	# Building out the "Information" page
	if selection == "Information":
		st.write("### General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("This application requires the user to input text (ideally a tweet relating to climate change), and will classify it according to whether or not they believe inclimate change.Below you will find information about the data source and a brief data description. You can have a look at word clouds and other general EDA on the EDA page, and make your predictions on the prediction page that you can navigate to in the sidebar.")
		st.markdown('---')
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Data Description'):
			st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).\nEach tweet is labelled as one of the following classes: \n- 2(News): the tweet links to factual news about climate change \n- 1(Pro): the tweet supports the belief of man-made climate change \n- 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change \n- -1(Anti): the tweet does not believe in man-made climate change")
			st.markdown('---')
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page for single input
	if selection == "Predict Single Input":
		up = Image.open('Images/Thumb_up2.png')
		down = Image.open('Images/Thumb_down2.png')
		neutral = Image.open('Images/neutral2.png')
		nerd = Image.open('Images/nerd2.png')
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		tweet_text = preprocess(tweet_text)
		choice = st.selectbox('Choose_model', ['Logistic Regression', 'Logistic Regression With Resampling', 'Naive Bayes'])
		if choice == 'Logistic Regression With Resampling':
			if st.button("Submit"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/model_lr_s.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [-1]:
					st.success("This text is by an ANTI-climate change")
					st.image(down, use_column_width=True)
				elif prediction == [1]:
					st.success("This text is by a PRO (A believer of climate change)")
					st.image(up, use_column_width=True)
				elif prediction == [2]:
					st.success("This text is by an NEWS type (Strictly follows facts of issues)")
					st.image(nerd, use_column_width=True)
				elif prediction == [0]:
					st.success('This text is by a NEUTRAL person')
					st.image(neutral, use_column_width=True)

		if choice == 'Logistic Regression':
			if st.button("Submit"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/model_lr_max.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [-1]:
					st.success("This text is by an ANTI-climate change")
					st.image(down, use_column_width=True)
				elif prediction == [1]:
					st.success("This text is by a PRO (A believer of climate change)")
					st.image(up, use_column_width=True)
				elif prediction == [2]:
					st.success("This text is by an NEWS type (Strictly follows facts of issues)")
					st.image(nerd, use_column_width=True)
				elif prediction == [0]:
					st.success('This text is by a NEUTRAL person')
					st.image(neutral, use_column_width=True)

		if choice == 'Naive Bayes':
			if st.button("Submit"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/model_nb_max.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [-1]:
					st.success("This text is by an ANTI-climate change")
					st.image(down, use_column_width=True)
				elif prediction == [1]:
					st.success("This text is by a PRO (A believer of climate change)")
					st.image(up, use_column_width=True)
				elif prediction == [2]:
					st.success("This text is by an NEWS type (Strictly follows facts of issues)")
					st.image(nerd, use_column_width=True)
				elif prediction == [0]:
					st.success('This text is by a NEUTRAL person')
					st.image(neutral, use_column_width=True)

			
			
	#Building out Prediction page for csv file
	if selection == "Predict CSV":
		st.subheader("Predict CSV")
		uploaded_file = st.file_uploader("Upload CSV", type='csv')
		if uploaded_file is not None:
			data = pd.read_csv(uploaded_file)
			#Display data
			st.subheader("Uploaded file")
			st.write(data)
			data = data.dropna()
			data.columns = ['Tweets', 'TweetID']
			data['Clean_Tweets'] = data['Tweets'].apply(preprocess)
			data['message_length'] = data['Clean_Tweets'].apply(word_length)
			if st.checkbox("Show Tweets Lengths Chart"):
				length_graph(data)
			if st.checkbox('Predict your data'):
				vect_data = tweet_cv.transform(data['Clean_Tweets'].values).toarray()
				model_choice = st.selectbox('Choose_model', ['Logistic Regression With Resampling', 'Logistic Regression', 'Naive Bayes'])
				if model_choice == 'Logistic Regression':
					predictor = joblib.load(open(os.path.join("resources/model_lr_max.pkl"),"rb"))
					prediction = predictor.predict(vect_data)
					st.markdown('### Prediction Results')
					output_df = pd.DataFrame({'Sentiments':prediction})
					full_df = output_df.join(data)
					mapping = {2:'News',1:'Pro',0:'Neutral',-1:'Anti'}
					full_df['Sentiments'] = full_df['Sentiments'].map(mapping)
					full_df_2 = full_df[["Tweets", "Sentiments"]]
					if st.checkbox('Show your predicted data'):
						st.write(full_df_2)
					with st.expander("# Explore Your Data And Predictions"):
						if st.checkbox('Who Is Dominating Your Data'):
							dominate(full_df)
						if st.checkbox('Show Distribution of Tweet Lenghts For Each Class'):
							length_graph_class(full_df)
						if st.checkbox("View Average Length Of Tweets Per Sentiment"):
							avgs = avg_lenght_df(full_df)
							st.write(avgs)
						if st.checkbox('View The Most Tweeted Words By Sentiment'):
							pro, news, neutral, anti = classes(full_df)
							if st.checkbox('Pro'):
								top_words(pro, title_class = "Pro")
							if st.checkbox('Anti'):
								top_words(anti, title_class = "Anti")
							if st.checkbox('News'):
								top_words(news, title_class = "News")
							if st.checkbox('Neutral'):
								top_words(pro, title_class = "Neutral")
					if st.checkbox("Download Output"):
						csv = full_df_2.to_csv(index=False)  # Convert DataFrame to CSV
						b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV as base64
						href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
						st.markdown(href, unsafe_allow_html=True)

				if model_choice == 'Logistic Regression With Resampling':
					predictor = joblib.load(open(os.path.join("resources/model_lr_s.pkl"),"rb"))
					prediction = predictor.predict(vect_data)
					st.markdown('### Prediction Results')
					output_df = pd.DataFrame({'Sentiments':prediction})
					full_df = output_df.join(data)
					mapping = {2:'News',1:'Pro',0:'Neutral',-1:'Anti'}
					full_df['Sentiments'] = full_df['Sentiments'].map(mapping)
					full_df_2 = full_df[["Tweets", "Sentiments"]]
					if st.checkbox('Show your predicted data'):
						st.write(full_df_2)
					with st.expander("# Explore Your Data And Predictions"):
						if st.checkbox('Who Is Dominating Your Data'):
							dominate(full_df)
						if st.checkbox('Show Distribution of Tweet Lenghts For Each Class'):
							length_graph_class(full_df)
						if st.checkbox("View Average Length Of Tweets Per Sentiment"):
							avgs = avg_lenght_df(full_df)
							st.write(avgs)
						if st.checkbox('View The Most Tweeted Words By Sentiment'):
							pro, news, neutral, anti = classes(full_df)
							if st.checkbox('Pro'):
								top_words(pro, title_class = "Pro")
							if st.checkbox('Anti'):
								top_words(anti, title_class = "Anti")
							if st.checkbox('News'):
								top_words(news, title_class = "News")
							if st.checkbox('Neutral'):
								top_words(pro, title_class = "Neutral")
					if st.checkbox("Download Output"):
						csv = full_df_2.to_csv(index=False)  # Convert DataFrame to CSV
						b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV as base64
						href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
						st.markdown(href, unsafe_allow_html=True)

				if model_choice == 'Naive Bayes':
					predictor = joblib.load(open(os.path.join("resources/model_nb_max.pkl"),"rb"))
					prediction = predictor.predict(vect_data)
					st.markdown('### Prediction Results')
					output_df = pd.DataFrame({'Sentiments':prediction})
					full_df = output_df.join(data)
					mapping = {2:'News',1:'Pro',0:'Neutral',-1:'Anti'}
					full_df['Sentiments'] = full_df['Sentiments'].map(mapping)
					full_df_2 = full_df[["Tweets", "Sentiments"]]
					if st.checkbox('Show your predicted data'):
						st.write(full_df_2)
					with st.expander("# Explore Your Data And Predictions"):
						if st.checkbox('Who Is Dominating Your Data'):
							dominate(full_df)
						if st.checkbox('Show Distribution of Tweet Lenghts For Each Class'):
							length_graph_class(full_df)
						if st.checkbox("View Average Length Of Tweets Per Sentiment"):
							avgs = avg_lenght_df(full_df)
							st.write(avgs)
						if st.checkbox('View The Most Tweeted Words By Sentiment'):
							pro, news, neutral, anti = classes(full_df)
							if st.checkbox('Pro'):
								top_words(pro, title_class = "Pro")
							if st.checkbox('Anti'):
								top_words(anti, title_class = "Anti")
							if st.checkbox('News'):
								top_words(news, title_class = "News")
							if st.checkbox('Neutral'):
								top_words(pro, title_class = "Neutral")
					if st.checkbox("Download Output"):
						csv = full_df_2.to_csv(index=False)  # Convert DataFrame to CSV
						b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV as base64
						href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
						st.markdown(href, unsafe_allow_html=True)

	#Building out the Explore page
	n_cloud= Image.open('Images/neutral cloud.png')
	ns_cloud= Image.open('Images/news_cloud.png')
	p_cloud= Image.open('Images/pro_cloud.png')
	a_cloud= Image.open('Images/anti_cloud.png')
	bar = Image.open('Images/bar.png')
	tweet_length = Image.open("Images/tweet length dist.png")
	tweet_length_classes = Image.open('Images/tweet length dist per class.png')
	word_frequency_pro = Image.open('Images/Pros.png')
	word_frequency_anti = Image.open('Images/Antis.png')
	word_frequency_news = Image.open('Images/News.png')
	word_frequency_neutral = Image.open('Images/Neutrals.png')
	if selection == "Explore":
		st.write('### Explore Data Statistics')
		st.markdown('---')
		with st.expander('# What tweeters are tweeting about climate change'):
			if st.checkbox('The Pro(Believers of Climate Change)'):
				st.image(p_cloud, use_column_width=True)
			if st.checkbox('The Anti(Non-believers of climate change)'):
				st.image(a_cloud, use_column_width=True)
			if st.checkbox('The News type(Relies of facts)'):
				st.image(ns_cloud, use_column_width=True)
			if st.checkbox('The Neutral(Have no stance)'):
				st.image(n_cloud, use_column_width=True)

		with st.expander('# Who is dominating our data'):
			st.image(bar, use_column_width=True)

		with st.expander('# Tweet Length Distribution'):
			st.markdown('The histogram below provides insights into the tweet lengths of the users in our sample dataset. It indicates that the majority of users have tweets ranging from 125 to 150 characters in length.')
			st.image(tweet_length, use_column_width=True)

		with st.expander('# Tweet Length Distribution Per Class'):
			st.markdown('Here below, the tweet length distribution has been displayed according to each class. They indicate the distribution of tweet lengths for each class in which majority of the Pros are texting within 100 to 150 wcharacters and majority of the other classes, as long as 100 to 140 characters')
			st.image(tweet_length_classes)

		with st.expander("# Most Tweeted Words by Sentiment Types"):
			st.markdown('The barcharts above show us the most used words by members of each of the sentimental classes.')
			if st.checkbox('Pro'):
				st.image(word_frequency_pro, use_column_width=True)
			if st.checkbox('Anti'):
				st.image(word_frequency_anti, use_column_width=True)
			if st.checkbox('News'):
				st.image(word_frequency_news, use_column_width=True)
			if st.checkbox('Neutral'):
				st.image(word_frequency_neutral, use_column_width=True)


	#Building out Documentation Page
	if selection == 'Documentation':
		st.write('### App Documentation')
		st.markdown('---')
		with st.expander('# Introduction'):
			st.markdown('Welcome to the documentation for the Climate Belief classification web app. This app provides a powerful tool for making predictions about an individual\'s belief in climate change based on their opinions and sentiments related to the topic. By leveraging natural language processing and machine learning techniques, the Climate Belief app aims to analyze and classify textual data to determine if a person believes in climate change or not.')
		
		with st.expander('# KeyFeatures'):
			st.markdown('**Sentiment Analysis:** The GreenGraph climate belief app employs advanced sentiment analysis algorithms to interpret the sentiments expressed in text data. By understanding the emotional tone, attitudes, and opinions conveyed in the input, it facilitates accurate prediction of climate change beliefs.')
			st.markdown('**Machine Learning Models:** The library incorporates a trained machine learning model specifically designed for climate change belief prediction. This model has been trained on a diverse dataset and can provide reliable predictions based on various inputs.')
			st.markdown('**No Installation required:** The app offers a simple and straightforward usage on the internet without any need for installation and its requiremnts, making it effortless to usw on all devices. By following the provided guidelines, you can seamlessly incorporate climate change belief predictions into your workflow.')
		
		with st.expander('# Benefits Of The Climate Belief App'):
			st.markdown('**Data-Driven Decision Making:** By utilizing GreenGraph, you can make data-driven decisions and gain insights into public opinion on climate change. This can be valuable for researchers, policymakers, and organizations seeking to understand public sentiment and tailor their strategies accordingly.')
			st.markdown('**Efficient Resource Allocation:** The ability to predict climate change beliefs can help allocate resources effectively. Organizations focused on climate advocacy, education, or public outreach can utilize Climate Belief to identify target audiences or allocate resources to regions where climate change skepticism may be more prevalent.')
			st.markdown('**Improved Communication and Understanding:** GreenGraph can contribute to fostering better communication and understanding between individuals with different perspectives on climate change. By recognizing the beliefs of others, it can help facilitate constructive discussions, empathy, and shared understanding.')

		with st.expander('# Usage'):
			st.markdown('This app provides the choice of either predicting single lines of tweet text or csvs containing several lines of tweets. We have integrated different model types with different capabailities for users to choose from.')
			st.markdown('Predicting single lines of tweets is as simple as navigating to the prediction tab in the navigation list of the sidepane, typing in the tweets in the text box following by choosing your choice of model and hitting \"submit". In a few seconds your prediction will be out and ready in an understandably and interpretable format.')
			st.markdown('In cases where it is necessary to predict bulk data, in the form of csv, all that is needed to be done is navigate to the \"Predict CSV", upload the csv, select the \"Predict Your Data" checkbox, choose the model of your choice, select the \"Download Output" checkbox and select the link the appears. With these steps followed the output containing the data that was fed in with an additional column with the predictions will be downloaded onto your device in csv format. **Note: Your input should be in csv format and strictly should contain two columns (One column containing the tweets or texts and the second containing the IDs for each row.**)')
			st.markdown('You can as well have a quick peek of a brief statistics of your data on the page. Just select the \"Explore Your Data and Predictions" expander and choose from the options that appear. You can also select the \"Show tweets length charts" checkbox to view a histogram of the length of tweets in your input')

		with st.expander('# You May Be Wondering...'):
			if st.checkbox(' What type of textual input does the app support?'):
				st.markdown('GreenGraph can process a wide range of textual input, including news articles, social media posts, blog posts, forum discussions, and any other form of text that expresses opinions or sentiments about climate change.')
			if st.checkbox('How accurate are the predictions made by GreenGraph?'):
				st.markdown('The accuracy of predictions depends on various factors, including the quality of the training data, the diversity of the input, and the specific context in which the predictions are made. ClimateBelief strives to provide reliable and accurate predictions, but it\'s important to note that no prediction model is 100%\ accurate. It is recommended to evaluate the performance of the model in your specific application or dataset.')
			if st.checkbox('Are there any limitations or considerations when using GreenGraph?'):
				st.markdown('While GreenGraph provides valuable insights, it\'s important to consider the limitations of any prediction model. ClimateBelief relies on the information and sentiments expressed in the text, which means it may not account for other factors that influence an individual\'s beliefs, such as personal experiences or background. Additionally, the predictions are based on probabilistic models and may not always align perfectly with individual beliefs. It\'s advisable to use ClimateBelief as a tool to augment understanding rather than relying solely on its predictions.')
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

