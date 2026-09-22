import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')



def plot_pie(dataframe, labels, sen_col):
    
    colors = ['mediumturquoise', 'gold', 'darkorange']
    unq_val = dataframe[sen_col].unique().tolist()
    values = [dataframe[sen_col].value_counts()[i] for i in unq_val]

    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=values)])

    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='value', 
                      textfont_size=20,
                      marker=dict(colors=colors, 
                                  line=dict(color='#000000', 
                                            width=2)))

    # fig.update_layout(autosize=False,
    #                   # width=400,
    #                   # height=400
    #                   )

    fig.show()

def plot_Neuneg_pie(dataframe):
    
    colors = ['gold', 'darkorange']

    fig = go.Figure(data=[go.Pie(labels=['Neutral','Negative'],
                                 values=[dataframe['Sentiment'].value_counts()[0],
                                         dataframe['Sentiment'].value_counts()[-1]],)])

    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='value', 
                      textfont_size=20,
                      marker=dict(colors=colors, 
                                  line=dict(color='#000000', 
                                            width=2)))

    # fig.update_layout(autosize=False,
    #                   # width=400,
    #                   # height=400
    #                   )

    fig.show()

def plot_Posneg_pie(dataframe):
    
    colors = ['mediumturquoise', 'darkorange']

    fig = go.Figure(data=[go.Pie(labels=['Positive','Negative'],
                                 values=[dataframe['Sentiment'].value_counts()[1],
                                 dataframe['Sentiment'].value_counts()[-1]],)])

    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='value', 
                      textfont_size=20,
                      marker=dict(colors=colors, 
                                  line=dict(color='#000000', 
                                            width=2)))

    # fig.update_layout(autosize=False,
    #                   # width=400,
    #                   # height=400
    #                   )

    fig.show()


def plot_Posneu_pie(dataframe):
    
    colors = ['mediumturquoise', 'gold']

    fig = go.Figure(data=[go.Pie(labels=['Positive','Neutral'],
                                 values=[dataframe['Sentiment'].value_counts()[1],
                                         dataframe['Sentiment'].value_counts()[0]],)])

    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='value', 
                      textfont_size=20,
                      marker=dict(colors=colors, 
                                  line=dict(color='#000000', 
                                            width=2)))

    # fig.update_layout(autosize=False,
    #                   # width=400,
    #                   # height=400
    #                   )

    fig.show()

    
    
def plot_dist(data, txt_col, sen_col, labels, bin_size):

    token_lengths = []

    for txt in data[txt_col]:
        tokens = tokenizer.encode(txt, 
                                  max_length=512,
                                  truncation=True)
        token_lengths.append(len(tokens))

    dataframe = data.copy(deep=True)
    dataframe['Length'] = token_lengths
    
    # x1 = dataframe[dataframe.Sentiment == 1].Length
    # x2 = dataframe[dataframe.Sentiment == 0].Length
    # x3 = dataframe[dataframe.Sentiment == -1].Length
    unq_val = dataframe[sen_col].unique().tolist()
    hist_data = [dataframe[dataframe[sen_col] == i].Length for i in unq_val]

    group_labels = labels
    color_set = ['#21C49C', '#FC8472', '#D45124']
    colours = [color_set[c] for c in range(len(unq_val))]
    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, 
                             group_labels, 
                             colors=colours,
                             bin_size=bin_size, 
                             show_rug=False)

    # Add title
    fig.update_layout(title_text='Hist and Curve Plot',
                      # autosize=False,
                      # width=1000, 
                      # height=500
                     )
    fig.show()




def plot_cm(y_test, y_pred, n_classes):

    def show_confusion_matrix(confusion_matrix, n_classes):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment');


    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=n_classes, columns=n_classes)
    show_confusion_matrix(df_cm, n_classes)




def plot_confusion_matrix(y_test, y_pred, labels, title):
  """
    cm : confusion matrix list(list)
    labels : name of the data list(str)
    title : title for the heatmap

  """

  cm = confusion_matrix(y_test, y_pred)
  data = go.Heatmap(z=cm, y=labels, x=labels, colorscale='Teal')
  annotations = []
  for i, row in enumerate(cm):
      for j, value in enumerate(row):
          annotations.append(
              {
                  "x": labels[i],
                  "y": labels[j],
                  "font": {"color": "yellow"},
                  "text": str(value),
                  "xref": "x1",
                  "yref": "y1",
                  "showarrow": False
              }
          )
  layout = {
      "title": title,
      "xaxis": {"title": "Predicted value"},
      "yaxis": {"title": "Real value"},
      "annotations": annotations
  }
  fig = go.Figure(data=data, layout=layout)
  fig.show()
