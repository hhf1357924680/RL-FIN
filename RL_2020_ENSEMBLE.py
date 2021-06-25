import warnings  # python运行代码的时候，经常会碰到代码可以正常运行但是会提出警告，不想看到这些不重要的警告，所以使用控制警告输出

warnings.filterwarnings("ignore")  # 使用警告过滤器来控制忽略发出的警告

import pandas as pd
import numpy as np
import matplotlib  # python中类似于MATLAB的绘图工具，是一个2D绘图库
import matplotlib.pyplot as plt
import datetime  # datetime模块提供了各种类,用于操作日期和时间


# %matplotlib inline 表示内嵌绘图，有了这个命令就可以省略掉plt.show()命令了
from finrl.config import config  # 引入finrl包的配置
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent, DRLEnsembleAgent

from finrl.trade.backtest import (
    backtest_stats,
    get_daily_return,
    get_baseline,
    backtest_plot,
)
from pprint import pprint  # 用于打印 Python 数据结构. 使输出数据格式整齐, 便于阅读

import sys  # 该语句告诉Python，我们想要使用sys，此模块包含了与Python解释器和它的环境有关的函数

sys.path.append("../FinRL-Library")
# 在Python执行import sys语句的时候，python会根据sys.path的路径来寻找sys.py模块。
# 添加自己的模块路径， Sys.path.append(“mine module path”)

import itertools  # itertools模块中的函数可以用来对数据进行循环操作

"""
os.path.exists(path)，如果path是一个存在的路径，返回True，否则返回 False
os.path.exists(path)的应用：判断路径是否存在，不存在则创建
举例子
log_dir = "logs/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
"""

import os

if not os.path.exists("./" + config.DATA_SAVE_DIR):  # "./"代表当前目录
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

# 下载数据
# Attributes
# ----------
#   start_date : str
#       start date of the data (modified from config.py)
#   end_date : str
#       end date of the data (modified from config.py)
#    ticker_list : list
#       a list of stock tickers (modified from config.py)

# Methods
# -------
# fetch_data()
#   Fetches data from yahoo API
# from config.py start_date is a string

config.START_DATE
config.END_DATE
print(config.DOW_30_TICKER)

# 缓存数据，如果日期或者股票列表发生变化，需要删除该缓存文件重新下载
SAVE_PATH = "./datasets/20210616-12h19.csv"
if os.path.exists(SAVE_PATH):
    df = pd.read_csv(SAVE_PATH)
else:
    df = YahooDownloader(
        config.START_DATE,  #'2000-01-01',
        config.END_DATE,  # 2021-01-01，预计将改日期改为'2021-06-20'（今日日期）
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()  # DOW_30_TICKER)道琼斯30只股票
    df.to_csv(SAVE_PATH)

df.head()  # 最开始5条
df.tail()  # tail仅展示了最后五条数据
df.shape
df.sort_values(["date", "tic"]).head()  # ticker表示股票代码，e.g.AAPL是苹果的股票
"""
pandas中的sort_values()函数可以根据指定行、列的数据进行排序
#DataFrame.sort_values(by=‘##’-按照指定列、行排序,ascending=True-默认升序排列, inplace=False-默认不替换原来数据集, na_position=‘last’-默认缺失值位置为last最后一个)
#按照df数据集的['date','tic']两列排序，其余参数取默认值
"""
# 数据预处理
"""
features：4+2+1+1(turb)
1.1. Add technical indicators: MACD  RSI  cci  adx
1.2. user_defined_feature: stock prices, current holding shares
1.3. current balance(当前账户所有额)
2.   turbulence index. 
 FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

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
事实上，当崩盘发生的时候，根本没时间反应


Methods
    -------
    preprocess_data()
        main method to do the feature engineering
"""

tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30"]

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=tech_indicators,
    use_turbulence=True,
    user_defined_feature=False,
)
##使用finrl.preprocessing.preprocessors中的FeatureEngineer来对股价数据进行预处理

processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()  # 按照processed的"tic"列去重
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)  # 成一个固定频率的时间索引
combination = list(itertools.product(list_date, list_ticker))
"""
1.pandas.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None, **kwargs)
由于import pandas as pd,所以也可以写成pd.date_range（start=None, end=None）
该函数主要用于生成一个固定频率的时间索引，使用时必须指定start、end、periods中的两个参数值，否则报错。
2.df.astype('str') #改变整个df变成str数据类型
3.itertools.product(*iterables[, repeat]) # 对应有序的重复抽样过程
  itertools.product(a,b),将a,b元组中的每个分量依次乘开。
"""

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
"""1.  pd.DataFrame( 某数据集 ，index  ，columns ),给某数据集加上行名index和列名columns
       此处只有pd.DataFrame( 某数据集 ，columns )，第一列加列名date，第二列加列名tic.
   2.  merge(df1,df2,on='key',how)
   按照["date","tic"]为关键字链接，以左边的dataframe为主导，左侧dataframe取全部数据，右侧dataframe配合左边
"""

processed_full = processed_full[processed_full["date"].isin(processed["date"])]
# isin函数，清洗数据，删选过滤掉processed_full中一些行，processed_full新加一列['date']若和processed_full中的['date']不相符合，则被剔除
processed_full = processed_full.sort_values(["date", "tic"])

processed_full = processed_full.fillna(0)
# 对于processed_full数据集中的缺失值使用 0 来填充.
processed_full.sample(5)  # sample（）是random模块中的一个函数，即随机取五个样本展示


# 设计强化学习实验环境
# trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data
# according to the principle of time-driven simulation.
# action space:{-1,0,1}-{selling,holding,buying};
# {k,...,-1,0,1,...,k}-{number of shares to sell,number of shares to hold,number of shares to buy}


# The continuous action space needs to be normalized to [-1, 1],
# since the policy is defined on a Gaussian distribution,
# which needs to be normalized and symmetric.

stock_dimension = len(processed_full.tic.unique())
state_space = 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

"""
1.按照processed_full的"tic"列去重并计算个数
2.计算状态空间的维数
"""
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,  # Since in Indonesia the minimum number of shares per trx is 100, then we scaled the initial amount by dividing it with 100
    "buy_cost_pct": 0.001,  # IPOT has 0.1% buy cost
    "sell_cost_pct": 0.001,  # IPOT has 0.1% sell cost
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5,
}

# 使用DRL算法（validating 3 agents:A2C、PPO、DDPG）
rebalance_window = 25  # rebalance_window is the number of days to retrain the model
validation_window = (
    25  # validation_window is the number of days to do validation and trading
)
# e.g. if validation_window=63, then both validation and trading period will be 63 days
train_start = "2017-01-01"
train_end = "2020-07-01"
val_test_start = "2020-07-01"
val_test_end = "2021-01-01"

ensemble_agent = DRLEnsembleAgent(
    df=processed_full,
    train_period=(train_start, train_end),
    val_test_period=(val_test_start, val_test_end),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs,
)

A2C_model_kwargs = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0005}

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

DDPG_model_kwargs = {
    "action_noise": "ornstein_uhlenbeck",
    "buffer_size": 50_000,
    "learning_rate": 0.000005,
    "batch_size": 128,
}

timesteps_dict = {"a2c": 30_000, "ppo": 100_000, "ddpg": 10_000}
# 疑问，为什么这里两个变量一样，赋值不一样呢？
timesteps_dict = {"a2c": 1_000, "ppo": 1_000, "ddpg": 1_000}


df_summary,model_ppo,model_a2c,model_ddpg = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict
)

models = [model_ppo,model_a2c,model_ddpg]

# r(s_t,a_t,s_(t+1) = (b_(t+1)+p_(t+1)*(h_(t+1)))-((b_t)+p_t*h_t)-ct
# 1.未使用model的累计收益
# 2.使用A2C得到的累计收益：A2C_model_kwargs
# 3.使用ppo得到的累计收益：PPO_model_kwargs
# 4.使用DDPG得到的累计收益：DDPG_model_kwargs
# 5.使用 集成策略 得到的累计收益：df_summary


def stat_result():
    # Backtest of Ensemble Strategy
    unique_trade_date = processed_full[
        (processed_full.date > val_test_start) & (processed_full.date <= val_test_end)
    ].date.unique()  # 使用划分好的验证集作为trade数据

    df_trade_date = pd.DataFrame(
        {"datadate": unique_trade_date}
    )  # 建立一个新的表，列名是datadate：内容是trade_date

    # 结果数据缓存在./result中，拼接所有的结果数据进行画图
    # ensemble, A2C, PPO, DDPG, (不使用策略)
    df_ensemble = pd.DataFrame() 
    for i in range(
        rebalance_window + validation_window,
        len(unique_trade_date) + 1,
        rebalance_window,
    ):
        temp = pd.read_csv(
            "results/account_value_trade_{}_{}.csv".format("ensemble", i)
        )
        df_ensemble = df_ensemble.append(temp, ignore_index=True)            

    sharpe = (
        (252 ** 0.5)
        * dfs[0].account_value.pct_change(1).mean()
        / dfs[0].account_value.pct_change(1).std()
    )
    print("Ensemble Sharpe Ratio: ", sharpe)
    dfs[0] = dfs[0].join(df_trade_date[validation_window:].reset_index(drop=True))

    #用3种模型去交易验证集中的数据
    ensemble_agent.DRL_validation(model=model_ddpg,test_data=validation,test_env=val_env_ddpg,test_obs=val_obs_ddpg)


    return dfs


dfs = stat_result()
# 将所有数据画到同一个图中
for i in [1,2,3]:
    x = dfs[i]["date"]
    y = dfs[i]["account_value"]
    #backtest_plot(dfs[i], '2020-07-02', '2020-11-20')
    plt.plot(x, y)

plt.savefig('account_value_plot.png')
