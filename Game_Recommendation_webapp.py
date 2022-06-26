import streamlit as ts
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time

def globalization(strok:str):
    return strok.replace(';', ' ')

# Removing duplicate tags
def del_rep(strok:str):
    spl_strok = strok.split()
    return ' '.join(sorted(set(spl_strok), key=spl_strok.index))

# Converting to lower case
def to_low(strok:str):
    return strok.lower()


def main():
        
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Game Recommendation System</h2>
    </div><br>
    """
    ts.markdown(html_temp, unsafe_allow_html=True)
    
    united_df = pd.read_csv('United_df.csv')   

    ts.markdown('<h3><b>Enter Game Name Here :</b></h3>', unsafe_allow_html=True)
    game = ts.text_input(label="")
    
    ts.markdown('<h3><b>Select number of games you want in recommendation</b></h3>', unsafe_allow_html=True)
    lst=[*range(1,21)]
    recs = ts.selectbox(label="",options=lst)
    
    # The main function that makes recommendations
    def get_rec(nam:str, cosine):
    #     Determine the index
        ind = united_df[united_df['name'] == nam].index.to_list()[0]

    #     Obtaining cosine convergence by index
        cos_scor = list(enumerate(cosine[ind]))

    #     Getting the most suitable games
        cos_scor = sorted(cos_scor, key=lambda x: x[1], reverse=True)
        cos_scor = cos_scor[1:(recs+1)]
        ten_ind = [i[0] for i in cos_scor]
        cand_s = united_df['name'].iloc[ten_ind]
        df = pd.DataFrame()
        new_df = df.assign(Name = cand_s)
        new_df.reset_index(drop=True, inplace = True)
        new_df.index = new_df.index + 1
        return new_df

    # Convert a collection of text documents to a matrix of token counts
    count_vec = CountVectorizer(stop_words='english')
    count_matrix = count_vec.fit_transform(united_df['alls'])

    # Convert a collection of raw documents to a matrix of TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(united_df['short_description'])

    # Stack sparse matrices horizontally
    st = hstack([count_matrix, tfidf_matrix])

    # Compute cosine similarity
    cosine_sim2 = cosine_similarity(st, st)
    
    ts.write('\n')
    #ts.write('Recommendations using game name, developer, genre, tags and description \n')
    if ts.button('Submit'):
        my_bar = ts.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete+1)
        ts.warning('These are the System Recommended Games:')
        ts.balloons()
        ts.write(get_rec(game, cosine_sim2),'\n')


                   
if __name__=='__main__':
                   main()
