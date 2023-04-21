import keras
import streamlit as st
from jobScrapper import JobScraper
from utilitiesFunction import *
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


def clear_text():
    st.session_state['url_input'] = ''
    st.session_state['title'] = ''
    st.session_state['company'] = ''
    st.session_state['department'] = ''
    st.session_state['industry'] = ''
    st.session_state['function'] = ''
    st.session_state['description'] = ''
    st.session_state['requirement'] = ''
    st.session_state['benefit'] = ''
    st.session_state['overview'] = ''


def preprocessInput(text):
    # Clean the text data
    textInput = basic_cleaning(text)
    tokens = tokenize_text(textInput)
    lemmaInput = lemmatize_token(tokens)
    stopwordFreeInput = remove_stopwords(lemmaInput)

    # Join the list of tokens back into a single string
    rejoined_text = ' '.join(stopwordFreeInput)

    # Preprocessed the text for LSTM
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([rejoined_text])
    X_seq = tokenizer.texts_to_sequences([rejoined_text])
    X_pad = pad_sequences(X_seq, maxlen=100)
    return X_pad


# Load the saved model from the HDF5 file
interfaceModel = keras.models.load_model('Model/tuned_LSTM.h5')


def url_page():
    st.title('Fake Job Posting Prediction with URL')

    # Define the URL to scrape
    url = st.text_input('Enter the JobStreet link to predict authenticity:', key='url_input')

    col1, col2 = st.columns([5, 1])
    with col1:
        if st.button('Predict Authenticity', key='button_for_URL'):
            if (is_jobstreet_url(url) == True):
                # call the JobScraper to scrape the job post
                scraper = JobScraper(url)
                job_data = scraper.scrape()

                # Call preprocessInput to preprocess the textual data for LSTM
                X_pad = preprocessInput(job_data)

                # Make a prediction on the preprocessed text
                prediction_prob = interfaceModel.predict(X_pad)
                prediction = prediction_prob[0][0]

                # Output the result
                if prediction > 0.5:
                    st.info('The job posting is fake.')
                else:
                    st.success('The job posting is authentic.')
                st.write(f"Probability of job post being fraud: {prediction:.2%}")
            else:
                st.error('Error: The given url should be a job street url.')

    with col2:
        clear = st.button('Clear Text', on_click=clear_text)


def form_page():
    employment = ['', 'Full-time', 'Contract', 'Part-time', 'Temporary', 'Other']
    education = ['', 'High School or Equivalent', 'Certification', "Bachelor's Degree", 'Professional',
                 'Associate Degree', 'Some College Coursework Completed', 'Some High School Coursework', 'Doctorate']
    experience = ['', 'Mid-Senior Level', 'Associate', 'Entry Level', 'Executive', 'Internship', 'Director',
                  'Not Applicable']
    st.title('Fake Job Posting Prediction with form submission')
    with st.form('jobPostForm'):
        jobTitle = st.text_input('Enter Job Title:', key='title')
        companyName = st.text_input('Enter Company Name:', key='company')
        department = st.text_input('Enter Company Department:', key='department')
        industry = st.text_input('Enter Industry:', key='industry')
        jobFunction = st.text_input('Enter Job Function:', key='function')
        jobDescription = st.text_input('Enter Job Description:', key='description')
        jobRequirement = st.text_input('Enter Job Requirement:', key='requirement')
        jobBenefits = st.text_input('Enter Job Benefits', key='benefit')
        companyOverview = st.text_input('Enter Company Overview:', key='overview')
        employmentType = st.selectbox('Select Employment Type', employment, index=0)
        requiredEducation = st.selectbox('Select Required Education', education, index=0)
        requiredExperience = st.selectbox('Select Required Experience', experience, index=0)
        hasLogo = st.radio('Does it have company logo?', ('Yes', 'No'), index=0)
        hasQuestion = st.radio('Does it have screening question?', ('Yes', 'No'), index=0)
        salaryRange = st.slider("Select the salary", min_value=0, max_value=50000, value=3000, step=50)

        col3, col4 = st.columns([5, 1])
        with col3:
            predictForm = st.form_submit_button(label='Predict Form')
        with col4:
            clearForm = st.form_submit_button(label="Clear Form", on_click=clear_text)

    if predictForm:
        # Validate required fields
        if not jobTitle.strip() or not companyName.strip() or not department.strip():
            st.error('Please enter a value for Job Title, Company Name, and Department.')
        else:
            form_data = {
                'jobTitle': jobTitle,
                'companyName': companyName,
                'department': department,
                'industry': industry,
                'jobFunction': jobFunction,
                'jobDescription': jobDescription,
                'jobRequirement': jobRequirement,
                'jobBenefits': jobBenefits,
                'companyOverview': companyOverview,
                'employmentType': employmentType,
                'requiredEducation': requiredEducation,
                'requiredExperience': requiredExperience,
                'hasLogo': hasLogo,
                'hasQuestion': hasQuestion,
                'salaryRange': salaryRange
            }
            form_data_str = ' '.join(str(value) for value in form_data.values())

            # Call preprocessInput to preprocess the textual data for LSTM
            X_pad1 = preprocessInput(form_data_str)

            # Make a prediction on the preprocessed text
            prediction_prob1 = interfaceModel.predict(X_pad1)
            prediction1 = prediction_prob1[0][0]

            # Output the result
            if prediction1 > 0.5:
                st.info('The job posting is fake.')
            else:
                st.success('The job posting is authentic.')
            st.write(f"Probability of job post being fraud: {prediction1:.2%}")


def tips():
    st.title('Tips to Avoid Job Fraud')
    st.markdown('---')

    with st.container():
        img_col, txt_col = st.columns([1, 2])
        with img_col:
            st.image('Images/company-research.jpg')
        with txt_col:
            st.subheader('Research the Company')
            st.write(
                'Before applying for a job, do some research on the company to ensure that it is a legitimate organization. Check their website, social media profiles, and read reviews from former employees on sites like Glassdoor. Look out for red flags such as poor reviews, lack of contact information, or inconsistent branding.')

    with st.container():
        img_col, txt_col = st.columns([1, 2])
        with img_col:
            st.image('Images/job-offer.png')
        with txt_col:
            st.subheader('Beware of Unsolicited Job Offers')
            st.write(
                'Be wary of unsolicited job offers that are sent to you via email or social media. Scammers often use these methods to lure unsuspecting job seekers into fake job opportunities.')

    with st.container():
        img_col, txt_col = st.columns([1, 2])
        with img_col:
            st.image('Images/fakeJobPost.jpg')
        with txt_col:
            st.subheader('Look Out for Fake Job Postings')
            st.write(
                'Job fraudsters may create fake job postings on legitimate job boards to attract applicants. Look for red flags such as poor grammar and spelling, lack of details about the job, or unrealistic salary expectations or validate with this web app.')

    with st.container():
        img_col, txt_col = st.columns([1, 2])
        with img_col:
            st.image('Images/neverPayForaJob.png')
        with txt_col:
            st.subheader('Never Pay for a Job')
            st.write(
                'Legitimate employers will never ask you to pay for a job or to pay for background checks or other fees. If a job requires payment, it is likely a scam.')

    with st.container():
        img_col, txt_col = st.columns([1, 2])
        with img_col:
            st.image('Images/checkAddress.jpg')
        with txt_col:
            st.subheader('Check for a Physical Address')
            st.write(
                'A legitimate company will have a physical address that you can verify. If the company does not provide an address, it may be a fake organization.')

    with st.container():
        img_col, txt_col = st.columns([1, 2])
        with img_col:
            st.image('Images/verifyJob.png')
        with txt_col:
            st.subheader('Verify the Job Offer')
            st.write(
                'Once you have applied for a job, verify the job offer before accepting it. A legitimate employer will provide you with a formal job offer letter that includes details about the job, salary, benefits, and start date.')


def app():
    st.set_page_config(page_title='Fake Job Prediction')
    pages = {
        'Url Prediction': url_page,
        'Form Prediction': form_page,
        'Tips to Avoid Job Fraud': tips
    }
    st.sidebar.title('Menu')
    page = st.sidebar.radio('Go to', tuple(pages.keys()))
    pages[page]()


app()
