import pandas as pd
import numpy as np
import joblib
import streamlit as st

# loading in the model to predict on the data
vect_path='./vectorizer.pkl'
model_path='./svm.pkl'

test_vect = joblib.load(vect_path)
test_model = joblib.load(model_path)

def welcome():
	return 'Hallo'

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(text):
	text = np.array([text], dtype=object)
	sample_text = test_vect.transform(text)
	result = test_model.predict(sample_text)

	print(result)
	return result

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("Analisis Sentimen Ulasan Aplikasi MyXL")
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:lightblue;padding:13px">
	<h4 style ="color:black;text-align:center;">Ulasan Aplikasi MyXL oleh Yayas Husni M & Rizki Noviyandi</h4>
	</div><br>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	review = st.text_input("Ulasan", "Ketik disini")
	result =""
	
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Prediksi"):
		result = prediction(review)[0]
	st.success('Hasilnya adalah {}'.format(result))
	
if __name__=='__main__':
	main()
