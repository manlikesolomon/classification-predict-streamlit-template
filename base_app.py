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
from nltk.corpus import stopwords


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
	options = ["Home", "Information", "Explore", "Prediction", 'Documentation']
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

	# Building out the predication page
	if selection == "Prediction":
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		tweet_text = preprocess(tweet_text)

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
			up = Image.open('Images/Thumb_up2.png')
			down = Image.open('Images/Thumb_down2.png')
			neutral = Image.open('Images/neutral2.png')
			nerd = Image.open('Images/nerd2.png')
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

	#Building out the Explore page
	n_cloud= Image.open('Images/neutral cloud.png')
	ns_cloud= Image.open('Images/news_cloud.png')
	p_cloud= Image.open('Images/pro_cloud.png')
	a_cloud= Image.open('Images/anti_cloud.png')
	bar = Image.open('Images/bar.png')

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
			st.markdown('The method of using the web app is as simple as navigating to the prediction tab in the navigation list of the sidepane, typing in the tweets in the text box and hitting \"submit\". In a few seconds your prediction will be out and ready in an understandably and interpretable format.')

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

