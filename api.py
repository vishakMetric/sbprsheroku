from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

print(app);

with open('pickel/model.sav','rb') as fp:
    logit_sm = pickle.load(fp)

with open('pickel/vectorizer.sav','rb') as vfp:
    word_vectorizer = pickle.load(vfp)
    
with open('pickel/userbased.sav','rb') as ufp:
    userbased = pickle.load(ufp)

with open('pickel/ratings.sav','rb') as rfp:
    rating = pickle.load(rfp)

with open('pickel/recommendation.sav','rb') as refp:
    recommendation = pickle.load(refp)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    username = request.form.get('Username')
    prediction = getPercentageDisplay(username)
    final_sorted_list = sorted(top5_dict.items(), key = lambda x:x[1], reverse = True)
    return render_template('index.html', topvalues=dict(final_sorted_list[0:5]))

#Method that will give the percentage of the top 5 products selected from previous input.
def getPositivePercentage(text_list, product, top5_dict):
    positive_pred = 0;
    negative_pred = 0;
    print("Into method getPositivePercentage");
    for i in range(len(text_list)):
        X_infer = word_vectorizer.transform([text_list[i]])
        pred = logit_sm.predict(X_infer)[0]
        if(pred == 1):
            positive_pred = positive_pred + 1
        else:
            negative_pred = negative_pred + 1
    total_prediction = positive_pred + negative_pred
    percentage = round((positive_pred/total_prediction) * 100 , 2)
    top5_dict[product] = percentage
    return top5_dict

def getPercentageDisplay(user_input):
    print("Into method getPercentageDisplay");
    d = userbased.loc[user_input].sort_values(ascending=False)[0:20]
    d = pd.merge(d,recommendation,left_on='name_id',right_on='name_id',how = 'left')
    unique_user = d['name_id'].unique()
    top_20_product_list = []
    for i in range(len(unique_user)):
        top_20_product_list.append(d[d['name_id'] == unique_user[i]]['name'].unique())

    top_20_product_df = pd.DataFrame(top_20_product_list, columns=["name"])
    list_of_text = []
    top5_dict = {}
    for count in range(len(top_20_product_df)):
        product = top_20_product_df.name[count]
        list_of_text = (rating[rating['name'] == top_20_product_df.name[count]]['reviews_text_clean']).tolist()
        getPositivePercentage(list_of_text, product, top5_dict)
        list_of_text = []
    

if __name__ == "__main__":
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port)
