import warnings#python运行代码的时候，经常会碰到代码可以正常运行但是会提出警告，不想看到这些不重要的警告，所以使用控制警告输出
warnings.filterwarnings("ignore")#使用警告过滤器来控制忽略发出的警告

import pandas as pd
import numpy as np
import matplotlib#python中类似于MATLAB的绘图工具，是一个2D绘图库
import matplotlib.pyplot as plt
#matplotlib.pyplot：提供一个类似于matlab的绘图框架（命令样式函数的集合）
#注：如果不希望显示绘图，则需要在，import matplotlib.pyplot as plt 之前使用 matplotlib.use('Agg')命令
# matplotlib.use('Agg')
import datetime#datetime模块提供了各种类,用于操作日期和时间

#%matplotlib inline# %matplotlib inline 表示内嵌绘图，有了这个命令就可以省略掉plt.show()命令了

from finrl.config import config#引入finrl包的配置
from finrl.marketdata.yahoodownloader import YahooDownloader#从finrl中引入YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent,DRLEnsembleAgent
from finrl.trade.backtest import backtest_stats, get_daily_return, get_baseline, backtest_plot

from pprint import pprint#用于打印 Python 数据结构. 使输出数据格式整齐, 便于阅读

import sys#该语句告诉Python，我们想要使用sys，此模块包含了与Python解释器和它的环境有关的函数
sys.path.append("../FinRL-Library")
#在Python执行import sys语句的时候，python会根据sys.path的路径来寻找sys.py模块。
#添加自己的模块路径， Sys.path.append(“mine module path”)

import itertools#itertools模块中的函数可以用来对数据进行循环操作

#os.path.exists(path)，如果path是一个存在的路径，返回True，否则返回 False
'''os.path.exists(path)的应用：判断路径是否存在，不存在则创建
举例子
log_dir = "logs/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
'''
import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):#"./"代表当前目录
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


#下载数据
# from config.py start_date is a string
config.START_DATE#更改此参数的方式：在config文件更改参数值后，刷新运行窗口并运行全部代码即可看到变化
config.END_DATE
print(config.DOW_30_TICKER)

#缓存数据，如果日期或者股票列表发生变化，需要删除该缓存文件重新下载
SAVE_PATH = "./datasets/20210616-12h19.csv"
if os.path.exists(SAVE_PATH):
    df = pd.read_csv(SAVE_PATH);
else:
    df = YahooDownloader(config.START_DATE,#'2000-01-01',
                        config.END_DATE,#预计将改日期改为'2021-06-20'（今日日期）
                        ticker_list = config.DOW_30_TICKER).fetch_data()
    df.to_csv(SAVE_PATH)


df.head()
df.tail()
df.shape
df.sort_values(['date','tic'],ignore_index=True).head()
#df.sort_values(['date','tic']).head()
'''pandas中的sort_values()函数可以根据指定行、列的数据进行排序
#DataFrame.sort_values(by=‘##’-按照指定列、行排序,ascending=True-默认升序排列, inplace=False-默认不替换原来数据集, na_position=‘last’-默认缺失值位置为last最后一个)
#按照df数据集的['date','tic']两列排序，其余参数取默认值
'''


#数据预处理
#使用finrl.preprocessing.preprocessors中的FeatureEngineer来对股价数据进行预处理
'''
Attributes
    ----------
        use_technical_indicator : boolean   注意：boolean(布尔值)取值为true or false,默认值是false
        we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean    
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not
注意：本文创新点是引入了一个turbulence提高模型的抗风险能力。文章通过定义turbulence的指数来反应股市状况，在崩盘的时候，则强制抛售所有资产以抵御股市崩盘的金融风险。
但是该特征比较trick，eg，崩盘事件概率太小，一般模型使用该指标合理么？且turbulence的阈值是超参数

Methods
    -------
    preprocess_data()
        main method to do the feature engineering'''

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=True,
    user_defined_feature = False)

processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()#按照processed的"tic"列去重
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))#成一个固定频率的时间索引
combination = list(itertools.product(list_date,list_ticker))
'''
1.pandas.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None, **kwargs)
由于import pandas as pd,所以也可以写成pd.date_range（start=None, end=None）
该函数主要用于生成一个固定频率的时间索引，使用时必须指定start、end、periods中的两个参数值，否则报错。
2.df.astype('str') #改变整个df变成str数据类型
3.itertools.product(*iterables[, repeat]) # 对应有序的重复抽样过程
  itertools.product(a,b),将a,b元组中的每个分量依次乘开。
'''

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
'''1.  pd.DataFrame( 某数据集 ，index  ，columns ),给某数据集加上行名index和列名columns
       此处只有pd.DataFrame( 某数据集 ，columns )，第一列加列名date，第二列加列名tic.
   2.  merge(df1,df2,on='key',how)
   按照["date","tic"]为关键字链接，以左边的dataframe为主导，左侧dataframe取全部数据，右侧dataframe配合左边
'''

processed_full = processed_full[processed_full['date'].isin(processed['date'])]
#isin函数，清洗数据，删选过滤掉processed_full中一些行，processed_full新加一列['date']若和processed_full中的['date']不相符合，则被剔除
processed_full = processed_full.sort_values(['date','tic'])
processed_full = processed_full.fillna(0)
#对于processed_full数据集中的缺失值使用 0 来填充.

processed_full.sample(5)
#对比原代码： processed_full.sort_values(['date','tic'],ignore_index=True).head(1)







#设计强化学习实验环境
#train = data_split(processed_full, '2009-01-01','2019-01-01')
#trade = data_split(processed_full, '2019-01-01','2021-01-01')

train = data_split(processed_full,'2017-01-01','2020-07-01')#training data split
#读取源文件夹‘processed_full’，并将其分割成train集合和trade集合
trade = data_split(processed_full,'2020-07-01','2021-01-01')#trade data split
print(len(train))
print(len(trade))
train.head()
trade.head()

config.TECHNICAL_INDICATORS_LIST#2018年的这篇文章有8个features
'''
'macd',
 'boll_ub'：布尔值上界；
 'boll_lb'：布尔值下界；
 'rsi_30',
 'cci_30',
 'dx_30',
 'close_30_sma',
 'close_60_sma'：收市价的60日简单移动平均:之前60天收市价的平均数
 '''

stock_dimension = len(train.tic.unique())#30支
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension#1+30*2+30*8
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
#stock_dimension=30，state_space=301
env_kwargs = {
    "hmax": 100, 
    "initial_amount":1000000, #Since in Indonesia the minimum number of shares per trx is 100, then we scaled the initial amount by dividing it with 100 
    "buy_cost_pct": 0.001, #IPOT has 0.1% buy cost
    "sell_cost_pct": 0.001, #IPOT has 0.1% sell cost
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
}
env_train_gym = StockTradingEnv(df = train, **env_kwargs)

env_train, _ = env_train_gym.get_sb_env()
print(type(env_train))

#使用DRL算法（validating 3 agents:A2C、PPO、DDPG）
#A2C
agent = DRLAgent(env = env_train)

model_a2c = agent.get_model("a2c")
trained_a2c = agent.train_model(model=model_a2c,
                               tb_log_name='a2c',
                               total_timesteps=100000)

#DDPG
agent = DRLAgent(env = env_train)
model_ddpg = agent.get_model("ddpg")

trained_ddpg = agent.train_model(model=model_ddpg,
                               tb_log_name='ddpg',
                               total_timesteps=50000)
#PPO
agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps":2048,
    "learning_rate":0.00025,
    "batch_size":128,
}
model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS)   

trained_ppo = agent.train_model(model=model_ppo,
                               tb_log_name='ppo',
                               total_timesteps=50000)

#TD3
agent = DRLAgent(env = env_train)
TD3_PARAMS = {
    "batch_size":100,
    "buffer_size":1000000,
    "learning_rate":0.001,
}
model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)  

trained_td3 = agent.train_model(model=model_td3,
                               tb_log_name='td3',
                               total_timesteps=30000)

#SAC
agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size":128,
    "buffer_size":1000000,
    "learning_rate":0.0001,
    "learning_starts":100,
    "ent_coef": "auto_0.1",
}
model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS) 
trained_sac = agent.train_model(model=model_sac,
                               tb_log_name='sac',
                               total_timesteps=80000)

#Trading
#Assume that we have $1,000,000 initial capital at 2019-01-01. We use the DDPG model to trade Dow jones 30 stocks.
#1.Set turbulence threshold(设置抗风险阈值)
data_turbulence = processed_full[(processed_full.date<'2020-07-01') & (processed_full.date>='2017-01-01')]
insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

insample_turbulence.turbulence.describe()
turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,1)
turbulence_threshold

#2.Trade
#DRL model needs to update periodically in order to take full advantage of the data, 
#ideally we need to retrain our model yearly, quarterly, or monthly.
#We also need to tune the parameters along the way, in this notebook I only use the in-sample data from 2009-01 to 2018-12 to tune the parameters once,
#so there is some alpha decay here as the length of trade date extends.
#Numerous hyperparameters – e.g. the learning rate, 
# the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

trade = data_split(processed_full, '2020-07-01','2021-01-01')
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 380, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()
trade.head()

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_sac, 
    environment = e_trade_gym)

df_account_value.shape

df_account_value.tail()
df_actions.head()

#测试算法策略
print("=======Get Backtest Results=======")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

#baseline stats
print("=======Get Baseline Stats=======")

baseline_df = get_baseline(
    ticker="^DJI",
    start = '2020-07-01',
    end = '2021-01-01')
    
stats = backtest_stats(baseline_df,value_col_name = 'close')


#backtest-plot
print("=======Comepare to DJIA=======")
#%matplotlib inline
#s&p 500:'GSPC
#Dow Jones Index: 'DJI
#NASDAQ 100: 'NDX
backtest_plot(df_account_value, 
             baseline_ticker = '^DJI', 
             baseline_start = '2020-07-01',
             baseline_end = '2021-01-01')

#缓存数据
SAVE_PATH = "./datasets/result.csv"
if os.path.exists(SAVE_PATH):
    df_account_value = pd.read_csv(SAVE_PATH);
else:
    df_account_value.to_csv(SAVE_PATH)