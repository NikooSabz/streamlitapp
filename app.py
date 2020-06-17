import streamlit as st 
import pandas as pd 
from datetime import date, timedelta, datetime
import numpy as np 
import plotly.express as px 
import math
import requests
from plotly import tools
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import random 
from countryinfo import CountryInfo
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from scipy import optimize
#from PIL import Image, ImageDraw, ImageFont


timely_response_url = 'PRI-TimelyResponse.csv'
st.title("PANDEMIC READINESS INDEX (PRI)")

st.cache(persist=True)
def load_data():
    data = pd.read_csv(timely_response_url) #.dropna(subset=['Num_Cases_Start_Lockdown'], inplace = True)
    data.drop(columns=['Place'], inplace = True)
    data.dropna(subset=['Num_Cases_Start_Lockdown'], inplace = True)
    data['End_Date'].replace(np.nan,  date.today()-timedelta(30), inplace = True)
    data['End_Date'] = pd.to_datetime(data['End_Date'])
    data['Start_Date'] = pd.to_datetime(data['Start_Date'])
    data['First_case_observed'] = pd.to_datetime(data['First_case_observed'])
    data['Length_Lockdown'] = (data['End_Date'] - data['Start_Date'])/np.timedelta64(1,'D') #.days()
    data['Countries_and_territories'] = data['Countries_and_territories'].str.strip()
    return data

data_ = load_data()

# Calculate Timely Response 
#data_['Timely_response'] = (data_['CasesofLockdown_UntilLockdown']/data_['Length_Lockdown']) 
data_['Timely_response'] = (data_['Since_Lockdown']/data_['Length_Lockdown'])
data_['Timely_response'] = 1 - (data_['Timely_response'] - min(data_['Timely_response'] ))/ \
                            (max(data_['Timely_response'])- min(data_['Timely_response'] ))


data_bar = data_[['Timely_response', 'Countries_and_territories']]
fig = px.bar(data_bar, x = "Countries_and_territories", y = "Timely_response", hover_data=['Countries_and_territories', 'Timely_response'])
#fig = px.bar(chart_data, x='countries', y='timely_rersponse', hover_data=['countries', 'timely_response'], height=400)
st.markdown("## Timely Responses")
st.write(fig)

if st.checkbox("Raw Data - Timely Response", False): 
    st.write(data_)


# Calculate Fiscal Response 
#st.markdown("## Fiscal Responses")

@st.cache(persist=True)
def data_fiscal_f():
  data_fiscal = pd.read_csv('PRI-FiscalResponse.csv')
  data_fiscal['Countries'] = data_fiscal['Countries'].str.strip()

  for i in ['1stround', '2ndround', '3rdround']:
    data_fiscal[i] = pd.to_datetime(data_fiscal[i])
  return data_fiscal

data_fiscal = data_fiscal_f() 

def exchange_rate_f(exchange_date, currency):

  URL = 'https://api.ratesapi.io/api/'+exchange_date.strftime("%Y-%m-%d")+'?base=USD&symbols='+ currency
  forx_currency = requests.get(URL).json()['rates'][currency]
  return float(1/forx_currency)

@st.cache(persist=True)
def fiscal_f(data_f, data_):
  fiscal_dic = {}
  dates_dic = {}
  cases_dic = {}

  for i in range(len(data_f)):
    fiscal = []
    dates = []
    cases = []
    k = 0
    for j in data_f.columns: 
      if ('fiscal' in j) and not math.isnan(data_f[j].iloc[i]):
        dates.append(data_f[data_f.columns[2+k]].iloc[i])
        
        if data_f['Currencies'].iloc[i] == "ISK" :
          exchange_rate = 0.0077
        else: 
          exchange_rate =  exchange_rate_f(data_f[data_f.columns[2+k]].iloc[i], data_f['Currencies'].iloc[i].upper())
        investment = data_f[j].iloc[i] * exchange_rate
        fiscal.append(investment)

        cases_date = data_f[data_f.columns[2+k]].iloc[i]
        cases_json = requests.get('https://corona.lmao.ninja/v2/historical/'+data_f['Countries'].iloc[i]+'?lastdays=300').json()
        cases.append(cases_json['timeline']['deaths'][str(cases_date.month)+'/'+str(cases_date.day)+'/'+str(cases_date.year)[-2:]])
        
        k += 1

    yesterday = date.today() - timedelta(days = 2)
    cases.append(cases_json['timeline']['deaths'][str(yesterday.month)+'/'+str(yesterday.day)+'/'+str(yesterday.year)[-2:]])

    fiscal_dic[data_f['Countries'].iloc[i]] = fiscal
    dates_dic[data_f['Countries'].iloc[i]] = dates
    cases_dic[data_f['Countries'].iloc[i]] = cases

  return fiscal_dic, dates_dic, cases_dic, cases

fiscal_dic, dates_dic, cases_dic, cases = fiscal_f(data_fiscal, data_)


fiscal_response = {}
for i in list(fiscal_dic.keys()): 

  j = 0
  pandemic_interval = (pd.to_datetime(date.today())-pd.to_datetime(data_['First_case_observed'][data_['Countries_and_territories'] == i]).iloc[0]).days

  cases_avg_grwth = cases_dic[i][0]/np.squeeze((dates_dic[i][0]-data_['First_case_observed'][data_['Countries_and_territories'] == i]).values/np.timedelta64(1,'D'))
  fiscal_avg_grwth = fiscal_dic[i][0]/np.squeeze((dates_dic[i][0]-data_['First_case_observed'][data_['Countries_and_territories'] == i]).values/np.timedelta64(1,'D'))

  for j in range(1, len(fiscal_dic[i])):
    fiscal_avg_grwth += (fiscal_dic[i][j])/((dates_dic[i][j] - dates_dic[i][j-1]).days)
    cases_avg_grwth += (cases_dic[i][j] - cases_dic[i][j-1])/((dates_dic[i][j] - dates_dic[i][j-1]).days)
    
  fiscal_avg_grwth /= len(fiscal_dic[i])
  cases_avg_grwth += (cases_dic[i][j+1] - cases_dic[i][j])/((pd.to_datetime(date.today()) - dates_dic[i][j]).days)
  cases_avg_grwth /= len(fiscal_dic[i]) + 1

  fiscal_response[i] = float(fiscal_avg_grwth/(pandemic_interval*cases_avg_grwth))


d = {'Countries': list(fiscal_response.keys()), 'Fiscal_response': list(fiscal_response.values())}
df_fiscal = pd.DataFrame(data=d)

df_fiscal['Fiscal_response'] = (df_fiscal['Fiscal_response']-min(df_fiscal['Fiscal_response']))/(max(df_fiscal['Fiscal_response']) - min(df_fiscal['Fiscal_response']))


fig = px.bar(df_fiscal, x = "Countries", y = "Fiscal_response", hover_data=['Countries', 'Fiscal_response'])
st.markdown("## Fiscal Responses")
st.write(fig)


if st.checkbox("Raw Data - Monetary Response", False): 
  st.write(data_fiscal)


# Other GHS dimenstions
st.markdown("## Other GHS Dimensions")

ghs_dims = pd.read_csv("PRI-GHS_dimensions.csv")

ghs_dims = ghs_dims[ghs_dims['Countries'].isin(list(data_fiscal['Countries']))]
for i in ghs_dims.columns[1:]: 
  ghs_dims[i] = (ghs_dims[i]-min(ghs_dims[i]))/(max(ghs_dims[i] - min(ghs_dims[i])))

colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(ghs_dims.columns))]
indx = 0
spider_data = []
for i in ghs_dims.columns[1:]:
  spider_data.append(go.Scatterpolar(
        name = i, #.Name.values[0],
        r = ghs_dims[i], #ghs_dims[a],
        theta = ghs_dims['Countries'], #['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
        fill = 'toself',
        line =  dict(
                color = colors[indx]
            )
        ))
  indx += 1

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 1]
    )
  ),
  showlegend = True,
)

fig = go.Figure(data=spider_data, layout=layout)

st.write(fig)

#df["Cases"] = round(df['Cumulative Cases'], 2)
Y = {}
population = {}
for i in data_fiscal['Countries']:
  Y[i] = cases_dic[i][-1]
  population[i] = CountryInfo(i).population()


d = {'Countries': list(Y.keys()), 'Deaths': list(Y.values()), 'Population': list(population.values())}
df = pd.DataFrame(data=d)

fig = go.Figure(data=[go.Scatter(
    x=df["Countries"], y=df['Population'],
    mode='markers',
    marker_size=df['Deaths']/1000,
)])

st.write(fig)


d = {'Countries': list(fiscal_response.keys()), 'Fiscal_response': list(fiscal_response.values())}
df_fiscal = pd.DataFrame(data=d)

df_fiscal['Fiscal_response'] = (df_fiscal['Fiscal_response']-min(df_fiscal['Fiscal_response']))/(max(df_fiscal['Fiscal_response']) - min(df_fiscal['Fiscal_response']))



model_dimensions = {'Countries': [], "fiscal_res": [], "timely_res": [], "PoE": [], "Risk_com": [],\
                    "Num_beds": [], "Gov_expenditure": [], "tourism_index": [], "Population_Density": []}
for i in list(df['Countries']):
  model_dimensions['Countries'].append(i)

  model_dimensions['fiscal_res'].append(round(df_fiscal['Fiscal_response'][df_fiscal['Countries']==i].values[0], 2))
  model_dimensions['timely_res'].append(round(data_['Timely_response'][data_['Countries_and_territories']==i].values[0], 2))
  model_dimensions['PoE'].append(round(ghs_dims['PoE'][ghs_dims['Countries']==i].values[0], 2))
  model_dimensions['Risk_com'].append(round(ghs_dims['Risk Communication'][ghs_dims['Countries']==i].values[0], 2))
  model_dimensions['Num_beds'].append(round(ghs_dims['Hospital beds'][ghs_dims['Countries']==i].values[0], 2))
  model_dimensions['Population_Density'].append(CountryInfo(i).population()/CountryInfo(i).area())
  model_dimensions['tourism_index'].append(round(ghs_dims['Tourism_Competitiveness_Index'][ghs_dims['Countries']==i].values[0], 2))
  model_dimensions['Gov_expenditure'].append(round(ghs_dims['Gov Expenditure'][ghs_dims['Countries']==i].values[0], 2))

model_dimensions_df = pd.DataFrame(data = model_dimensions)
model_dimensions_df['Population_Density'] = (model_dimensions_df['Population_Density']-min(model_dimensions_df['Population_Density']))/ \
                                  (max(model_dimensions_df['Population_Density'])- min(model_dimensions_df['Population_Density']))



## Fitted distributions 


'''
Plot parametric fitting.
'''
def utils_plot_parametric(dtf, zoom=30, figsize=(15,5)):
    ## interval
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    dtf["conf_int_low"] = dtf["forecast"] - 1.96*dtf["residuals"].std()
    dtf["conf_int_up"] = dtf["forecast"] + 1.96*dtf["residuals"].std()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    ## entire series
    dtf["ts"].plot(marker=".", linestyle='None', ax=ax[0], title="Parametric Fitting "+country, color="black")
    dtf["model"].plot(ax=ax[0], color="green")
    dtf["forecast"].plot(ax=ax[0], grid=True, color="red")
    ax[0].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)
    ax[0].legend(['Actual', 'Fitted/Estimated'])
    ## focus on last
    first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]
    first_loc = dtf.index.tolist().index(first_idx)
    zoom_idx = dtf.index[first_loc-zoom]
    dtf.loc[zoom_idx:]["ts"].plot(marker=".", linestyle='None', ax=ax[1], color="black", 
                                  title="Zoom on the last "+str(zoom)+" observations")
    dtf.loc[zoom_idx:]["model"].plot(ax=ax[1], color="green")
    dtf.loc[zoom_idx:]["forecast"].plot(ax=ax[1], grid=True, color="red")
    ax[1].fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['conf_int_low'], 
                       y2=dtf.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
    ax[1].legend(['Actual', 'Fitted/Estimated'])
    #plt.show()
    st.pyplot(fig)
    return dtf[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]


def forecast_curve(ts, f, model, pred_ahead=None, freq="D", zoom=60, figsize=(30,10)):
    ## fit
    X = np.arange(len(ts))
    fitted = f(X, model[0], model[1], model[2])
    dtf = ts.to_frame(name="ts")
    dtf["model"] = fitted
    
    ## index
    index = pd.date_range(start=str(datetime.date(datetime.now())),periods=pred_ahead,freq=freq)
    index = index[1:]
    ## forecast
    Xnew = np.arange(len(ts)+1, len(ts)+1+len(index))
    preds = f(Xnew, model[0], model[1], model[2])
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## plot
    utils_plot_parametric(dtf, zoom=zoom)
    return dtf

#'''
#Logistic function: f(x) = capacity / (1 + e^-k*(x - midpoint) ) ==> for cumulative cases 
#'''
def logistic_f(X, c, k, m):
    y = c / (1 + np.exp(-k*(X-m)))
    return y

#'''
#Gaussian function: f(x) = a * e^(-0.5 * ((x-μ)/σ)**2) ==> for daily cases (one peak only)
#'''
def gaussian_f(X, a, b, c):
    y = a * np.exp(-0.5 * ((X-b)/c)**2)
    return y


####################################
st.header("Estimate Covid Trends")
select = st.selectbox('Countries', list(model_dimensions_df['Countries']))

df= {'Countries': [], 'Response': [], 'Total_case': []}
for i in list(model_dimensions_df['Countries']): 
  if select == i: 

    cases_json = requests.get('https://corona.lmao.ninja/v2/historical/'+i+'?lastdays=300').json()

    dtf = pd.DataFrame(data = {'total': list(cases_json['timeline']['cases'].values()),
                              'new':  [0]+ list(np.ediff1d(np.asarray(list(cases_json['timeline']['cases'].values())))), \
                              'date': list(cases_json['timeline']['cases'].keys())})

    dtf['date'] = pd.to_datetime(dtf['date'].astype(str)) #, format='%Y/%m/%d')

    #dtf['date'].to.strftime('%Y/%m/%d')
    dtf.set_index('date', inplace=True)

    logistic_model, cov = optimize.curve_fit(logistic_f,\
                                  xdata=np.arange(len(dtf["total"])), \
                                  ydata=dtf["total"].values, \
                                  maxfev=100000,\
                                  p0=[np.mean(dtf["total"]), 1, 1])

    gaussian_model, cov = optimize.curve_fit(gaussian_f,
                                ydata=dtf["new"].values, 
                                xdata=np.arange(len(dtf["new"])),
                                maxfev=10000,
                                p0=[1, np.mean(dtf["new"]), np.std(dtf['new'])])

    country = i
    preds = forecast_curve(dtf["total"], logistic_f, logistic_model, pred_ahead=30, freq="D", zoom=7)

    preds = forecast_curve(dtf['new'], gaussian_f, gaussian_model, pred_ahead=20, freq="D", zoom=7)
    
#####################
#####################
d = {'Countries': [], 'Population': [], 'Case_growth': [], "Response": [], "Total_cases": []}

for i in list(ghs_dims['Countries']):
  cases_json = requests.get('https://corona.lmao.ninja/v2/historical/'+i+'?lastdays=300').json()

  length = (datetime.strptime(list(cases_json['timeline']['cases'].keys())[-1], '%m/%d/%y') - \
  datetime.strptime(list(cases_json['timeline']['cases'].keys())[0], '%m/%d/%y')).days

  dtf = pd.DataFrame(data = {'total': list(cases_json['timeline']['cases'].values()),
                            'new':  [0]+ list(np.ediff1d(np.asarray(list(cases_json['timeline']['cases'].values())))), \
                            'date': list(cases_json['timeline']['cases'].keys())})

  dtf['date'] = pd.to_datetime(dtf['date'].astype(str)) #, format='%Y/%m/%d')

  #dtf['date'].to.strftime('%Y/%m/%d')
  dtf.set_index('date', inplace=True)

  d['Case_growth'].append(sum(np.ediff1d(np.asarray(list(cases_json['timeline']['cases'].values()))))/length)
  logistic_model, cov = optimize.curve_fit(logistic_f,\
                                xdata=np.arange(len(dtf["total"])), \
                                ydata=dtf["total"].values, \
                                maxfev=100000,\
                                p0=[np.mean(dtf["total"]), 1, 1])
  a = []
  for j in range(300): 
    a.append(logistic_model[0] / (1 + np.exp(-logistic_model[1]*(j-logistic_model[2]))))
  #np.where(np.asarray(a) > (np.mean(a)-np.std(a)/np.sqrt(len(a))))
  b = np.ediff1d(np.asarray(a))

  d['Response'].append(max(np.where(np.asarray(b) > (np.mean(b)-np.std(b)/np.sqrt(len(b))))[0]))
  d['Total_cases'].append(dtf['total'][-1])
  d['Countries'].append(i)
  d['Population'].append(CountryInfo(i).population())

df = pd.DataFrame(data=d)
df['Y'] = df['Response']*df['Total_cases']/df['Population']

############
st.markdown("## Multivariate Linear Regression Analysis")

y_train = df['Response']
x_train = model_dimensions_df[model_dimensions_df.columns[1:]] #X.drop([df["Cases"].index[indx]]) #, inplace = True)
#x_train = x_train.drop(['PoE', 'Risk_com', 'Gov_expenditure'], axis=1)

ols = linear_model.LinearRegression()
model = ols.fit(x_train, y_train)

est = sm.OLS(y_train, sm.add_constant(x_train)).fit()

plt.clf()
plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(est.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png', bbox_inches='tight')
#image = Image.open('output.png')
#st.image(image)


Estimated_index = {"Country": [], "Score": []}

indx = 0
for i in model_dimensions_df['Countries']: 
  Estimated_index['Country'].append(i)
  Estimated_index['Score'].append(sum(np.multiply(model.coef_, x_train.iloc[indx])) + model.intercept_)

  indx += 1

Estimated_index = pd.DataFrame(data = Estimated_index)
Estimated_index['Score'] = 1 - (Estimated_index['Score']-min(Estimated_index['Score']))/ \
                            (max(Estimated_index['Score'])-min(Estimated_index['Score']))

fig = go.Figure(data=go.Choropleth(
    locations=Estimated_index['Country'], # Spatial coordinates
    z = Estimated_index['Score'].astype(float), # Data to be color-coded
    locationmode = 'country names', #'USA-states', # set of locations match entries in `locations`
    colorscale = 'viridis',
    #colorbar_title = i,
))

fig.update_layout(
    title_text = 'Cumulative # Covid-19 cases',
    geo_scope='world', # limite map scope to USA
    width=1000,
    height=800,
    legend = go.layout.Legend(
      x=0,
      y=1,
      traceorder="normal",
))

st.write(fig)