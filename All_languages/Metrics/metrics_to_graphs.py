import csv 
import os
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# assign directory
directory = '/home/tbidewell/home/POS_tagging/Data/Metrics'
 
# iterate over files in
# that directory
list_of_langs = []

ep_loss_tr = {}
ep_loss_dv = {}
ep_acc_tr = {}
ep_acc_dv = {}
test_loss = {}
test_acc = {}

# (w_lst, w_ch_lstm, trans)

def extract_data(lang_f, model_name):
    for file in os.listdir(model_path):
        if "epoch_losses_train.csv" in file:
            with open(model_path + '/epoch_losses_train.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                total = []
                for i in data:
                    if 'Loss' not in i[0]:
                        total.append(float(i[0][1:-1]))
                
                if lang_f in ep_loss_tr:
                    ep_loss_tr[lang_f].append(np.mean(total))
                else:
                    ep_loss_tr[lang_f] = [np.mean(total)]
        
        if "epoch_losses_dev.csv" in file:
            with open(model_path + '/epoch_losses_dev.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                total = []
                for i in data:
                    if 'Loss' not in i[0]:
                        total.append(float(i[0][1:-1]))

                if lang_f in ep_loss_dv:
                    ep_loss_dv[lang_f].append(np.mean(total))
                else:
                    ep_loss_dv[lang_f] = [np.mean(total)]

        if "epoch_accuracy_train.csv" in file:
            with open(model_path + '/epoch_accuracy_train.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                total = []
                for i in data:
                    if 'Accuracy' not in i[0]:
                        total.append(float(i[0][1:-1]))

                if lang_f in ep_acc_tr:
                    ep_acc_tr[lang_f].append(np.mean(total))
                else:
                    ep_acc_tr[lang_f] = [np.mean(total)]

        if "epoch_accuracy_dev.csv" in file:
            with open(model_path + '/epoch_accuracy_dev.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                total = []
                for i in data:
                    if 'Accuracy' not in i[0]:
                        total.append(float(i[0][1:-1]))

                if lang_f in ep_acc_dv:
                    ep_acc_dv[lang_f].append(np.mean(total))
                else:
                    ep_acc_dv[lang_f] = [np.mean(total)]

        if "test_accuracy_all.csv" in file:
            with open(model_path + '/test_accuracy_all.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                total = []
                for i in data:
                    if 'Accuracy' not in i[0]:
                        total.append(float(i[0][1:-1]))

                if lang_f in test_acc:
                    test_acc[lang_f].append(np.mean(total))
                else:
                    test_acc[lang_f] = [np.mean(total)]

        if "test_loss_all.csv" in file:
            with open(model_path + '/test_loss_all.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                total = []
                for i in data:
                    if 'Loss' not in i[0]:
                        total.append(float(i[0][1:-1]))

                if lang_f in test_loss:
                    test_loss[lang_f].append(np.mean(total))
                else:
                    test_loss[lang_f] = [np.mean(total)]

for lang in os.listdir(directory):
    lang_f = os.path.join(directory, lang)
    list_of_langs.append(lang_f.split("/")[-1])


    for model in os.listdir(lang_f):

        model_path = os.path.join(lang_f, model)

        if "w_ch_lstm" in model:
            l = lang_f.split("/")[-1]
            extract_data(l, "w_ch_lstm")
            
        
        if "w_lstm" in model:    
            l = lang_f.split("/")[-1]        
            extract_data(l, "w_lstm")

        if "transformer" in model:  
            l = lang_f.split("/")[-1]
            extract_data(l, "transformer")          
            
        

#word_lstm_ep_loss_tr = word_lstm_ep_loss_tr[:len(word_lstm_ep_loss_tr)-1]


def scatter_plot(df, title):
    fig = px.scatter(df, x= df.columns[1], y=df.columns[2], text='Treebank')
    #fig = px.scatter(df, x= df.columns[1], y=df.columns[2])
    fig.update_traces(textposition='top center')

    fig.update_layout(
        height=800,
        title_text= title
    )

    fig.show()


def get_df(name_1, list_1, name_2, list_2):
    return pd.DataFrame(list(zip(list_of_langs, list_1, list_2)),
               columns =['Treebank', name_1, name_2])


combi_dict = {"w_lstm": "0", 
              "w_ch_lstm": "1",
              "transformer": "2"}


def plot_graph(dict_of_vals, title, combination, text = False):

    id_1, id_2 = combination

    df = pd.DataFrame.from_dict(dict_of_vals, orient='index', columns=['w_lstm', 'w_ch_lstm', 'transformer'])


    if text:
        fig = px.scatter(df, x= df.columns[int(combi_dict[id_1])], y=df.columns[int(combi_dict[id_2])], text=df.index.tolist())
    else:
        fig = px.scatter(df, x= df.columns[int(combi_dict[id_1])], y=df.columns[int(combi_dict[id_2])])

    #fig = px.scatter(df, x= df.columns[1], y=df.columns[2])
    fig.update_traces(textposition='top center')

    fig.update_layout(
        height=800,
        title_text= title
    )

    fig.show()
    

plot_graph(ep_acc_dv, 'Dev Accuracy: w_ch_lstm vs w_lstm',  ("w_ch_lstm", "w_lstm"), text=True)
plot_graph(ep_acc_dv, 'Dev Accuracy: w_ch_lstm vs transformer',  ("w_ch_lstm", "transformer"), text=True)
plot_graph(ep_acc_dv, 'Dev Accuracy: w_lstm vs transformer',  ("w_lstm", "transformer"), text=True)



