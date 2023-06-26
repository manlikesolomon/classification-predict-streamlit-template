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


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
	
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#sets the background image
	with open('design.css') as designs:
		st.markdown(f"<style>{designs.read()}</style>", unsafe_allow_html=True)

	logo = Image.open('Images/white.png')
	#logo2 = Image.open('Images/gg.png')
	st.image(logo, use_column_width=True)
	#st.image(logo2)
	st.markdown(f'<img src="http://54.154.172.124:5000/media/6b4b9df26b5cb297db50888a170735f06acdcd490592b420e511ae81.png" alt="logo" class="logo-image">', unsafe_allow_html=True)
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	logo1 = Image.open('Images/robog.jpg')
	st.sidebar.image(logo1, use_column_width=True)
	options = ["Home", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)
	
	#building out Homepage
	if selection == "Home":
		st.markdown('# HOME')
		if st.checkbox('About US'):
			st.info("## Pioneers")
			st.markdown('---')
			st.markdown('Welcome to GreenGraph Solutions! We are not just a data science group; we are visionaries, innovators, and pioneers in the world of data-driven solutions. With a relentless passion for harnessing the power of data, we have embarked on a mission to shape the future through our cutting-edge predictive analytics capabilities. At GreenGraph Solutions, we believe that data holds the key to unlocking endless possibilities. We combine our expertise in machine learning, artificial intelligence, and advanced analytics to uncover hidden insights, solve complex problems, and drive informed decision-making. Our team of talented data scientists, engineers, and domain experts work tirelessly to develop customized solutions that empower businesses and organizations across industries. What sets us apart is our unwavering commitment to delivering excellence. We leverage the latest technologies and employ state-of-the-art methodologies to ensure the highest level of accuracy, reliability, and scalability in our predictions. Our track record of success speaks for itself as we have helped numerous clients achieve remarkable outcomes and gain a competitive edge in their respective fields. Beyond our technical prowess, we pride ourselves on our collaborative approach and client-centric mindset. We work closely with our partners, understanding their unique challenges and goals, to co-create tailored solutions that drive meaningful impact. Our dedication to building long-term relationships and providing exceptional customer service sets us apart as a trusted advisor and strategic partner. Join us on this exhilarating journey as we redefine what is possible in the world of data science. Let GreenGraph Solutions be your trusted guide as we navigate the vast landscape of data, transforming it into actionable insights and unlocking the true potential of your business. Together, let us shape a brighter, data-driven future. GreenGraph Solutions - Where Data Meets Possibilities.')
		

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

		if st.button("Submit"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == [-1]:
				st.success("This text is by an ANTI-climate change")
			elif prediction == [1]:
				st.success("This text is by a PRO (A believer of climate change)")
			elif prediction == [2]:
				st.success("This text is by an NEWS type (Strictly follows facts of issues)")
			elif prediction == [0]:
				st.success('This text is by a NEUTRAL person')
	
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

