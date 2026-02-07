import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

data = pd.read_csv('data/monthly_mocha.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)

plt.plot(data['date'], data['subscriptions'])
plt.savefig('models/depvar.png', dpi=300, bbox_inches='tight')
plt.close()

stl = STL(data['subscriptions'].values, seasonal=13, period = 4)
result = stl.fit()
mape = np.mean(np.abs(result.resid / data['subscriptions'].values)) * 100
mape

result.plot()
plt.savefig('models/stl_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()

control_data = pd.DataFrame({
    'trend': result.trend,
    'seasonal': result.seasonal,
})


data = pd.concat([data,control_data], axis = 1)
data['revenue_per_kpi'] = data['subscriptions'] * 100
data.to_csv('data/monthly_mocha_ctrls.csv')

data_train = data[:-4]
data_test = data[-4:]

data_train.to_csv('data/train_m_mocha.csv')
data_test.to_csv('data/test_m_mocha.csv')

target = "subscriptions"

date_var = "date"

media = ['meta_spend', 'google_spend', 'snapchat_spend', 
            'tiktok_spend', 'moloco_spend','liveintent_spend',
            'roku_spend','beehiiv_spend','amazon_spend']
controls = ["trend","seasonal"]

organic = ['meta_impressions', 'google_impressions', 'snapchat_impressions', 
            'tiktok_impressions', 'moloco_impressions','liveintent_impressions',
            'roku_impressions','beehiiv_impressions','amazon_impressions']

# Calcular correlación
correlations = []

for i, channel in enumerate(media):
    spend_col = channel  # e.g., 'meta_spend'
    impression_col = organic[i]  # e.g., 'meta_impressions'
    
    corr = data[[spend_col, impression_col]].corr().iloc[0, 1]
    
    correlations.append({
        'channel': channel.replace('_spend', ''),
        'correlation': corr
    })

df_corr = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
print(df_corr.to_string(index=False))