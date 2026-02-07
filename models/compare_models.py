import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from pymc_toolkit.pymc_model import PymcModel
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('data/monthly_mocha.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)

plt.plot(data['date'], data['subscriptions'])
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

data = pd.concat([data,control_data],axis =1)

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

####################################################
#           Models
####################################################

## Modelo Weibull media
model0 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=media,
             control_names=controls,
             lag_max=4,
             scale_data=True,
             adstock="weibull_pdf",
             saturation='michaelis_menten')

pfleet0 = model0.production_fleet(n_test=12)
pfleet0.summary()
az.loo(pfleet0.mmm.idata, var_name="y", pointwise=True)
pfleet0.generate_report(output_html="models/weibull_media_controls.html")

## Modelo Weibull Impressions
model01 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=organic,
             control_names=controls,
             lag_max=8,
             scale_data=True,
             adstock="weibull_pdf",
             saturation='michaelis_menten')

pfleet01 = model01.production_fleet(n_test=12)
pfleet01.summary()
az.loo(pfleet01.mmm.idata, var_name="y", pointwise=True)
pfleet01.generate_report(output_html="models/weibull_impressions_controls.html")

## modelo Geometric media n controls
model1 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=media,
             control_names=controls,
             lag_max=4,
             scale_data=True,
             adstock="geometric",
             saturation='michaelis_menten')

pfleet1 = model1.production_fleet(n_test=12)
pfleet1.summary()
az.loo(pfleet1.mmm.idata, var_name="y", pointwise=True)
pfleet1.generate_report(output_html="models/geometric_media_controls.html")

## modelo Geometric impressions n controls
model11 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=organic,
             control_names=controls,
             lag_max=4,
             scale_data=True,
             adstock="geometric",
             saturation='michaelis_menten')

pfleet11 = model11.production_fleet(n_test=12)
pfleet11.summary()
az.loo(pfleet11.mmm.idata, var_name="y", pointwise=True)
pfleet11.generate_report(output_html="models/geometric_impressions_controls.html")

## modelo weibull media no controls
model2 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=media,
             lag_max=4,
             scale_data=True,
             adstock="weibull_pdf",
             saturation='michaelis_menten')

pfleet2 = model2.production_fleet(n_test=12)
pfleet2.summary()
az.loo(pfleet2.mmm.idata, var_name="y", pointwise=True)
pfleet2.generate_report(output_html="models/weibull_media_no_controls.html")

## modelo weibull impressions no controls
model21 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=organic,
             lag_max=4,
             scale_data=True,
             adstock="weibull_pdf",
             saturation='michaelis_menten')

pfleet21 = model21.production_fleet(n_test=12)
pfleet21.summary()
az.loo(pfleet21.mmm.idata, var_name="y", pointwise=True)
pfleet21.generate_report(output_html="models/weibull_impressions_no_controls.html")

## modelo Geometric media_plus_impre vars and no controls
model3 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=media+organic,
             lag_max=4,
             scale_data=True,
             adstock="geometric",
             saturation='michaelis_menten')

pfleet3 = model3.production_fleet(n_test=12)
pfleet3.summary()
az.loo(pfleet3.mmm.idata, var_name="y", pointwise=True)
pfleet3.generate_report(output_html="models/geometric_media_plus_no_controls.html")

## modelo Geometric with media_plus and controls
model4 = PymcModel(client_data=data,
             target_name=target,
             depvar_type ="conversions",
             date_column=date_var,
             channel_names=media + organic,
             control_names=controls,
             lag_max=4,
             scale_data=True,
             adstock="geometric",
             saturation='michaelis_menten')

pfleet4 = model4.production_fleet(n_test=12)
pfleet4.summary()
az.loo(pfleet4.mmm.idata, var_name="y", pointwise=True)
pfleet4.generate_report(output_html="models/geometric_media_plus_controls.html")

################################
## compare predictive power
#################################

results = {}
for name, fleet in [
    ("weibull_media_controls", pfleet0),
    ("weibull_impressions_controls", pfleet0),
    ("geometric_media_controls", pfleet1),
    ("geometric_impressions_controls", pfleet11),
    ("weibull_media_no_controls", pfleet2),
    ("weibull_impressions_no_controls", pfleet21),
    ("geometric_media_plus_no_controls", pfleet3),
    ("geometric_media_plus_controls", pfleet4)
]:
    if fleet.type == "production":
        metrics = fleet.get_model_accuracy(train=False)
        results[name] = {
            'crps_error': metrics['crps_error'],
            'mape': metrics['mape']
        }

results = dict(sorted(results.items(), key=lambda x: x[1]['crps_error']))

df = pd.DataFrame(results).T
df = df.sort_values('crps_error').reset_index()
df = df.rename(columns={'index': 'model'})

################################
## compare media contributions
#################################

results = []
for name, fleet in [
    ("weibull_media_controls", pfleet0),
    ("weibull_impressions_controls", pfleet0),
    ("geometric_media_controls", pfleet1),
    ("geometric_impressions_controls", pfleet11),
    ("weibull_media_no_controls", pfleet2),
    ("weibull_impressions_no_controls", pfleet21),
    ("geometric_media_plus_no_controls", pfleet3),
    ("geometric_media_plus_controls", pfleet4)
]:
    contrib = fleet.mmm.compute_mean_contributions_over_time().sum()
    total = contrib.sum()
    
    media = contrib[fleet.mmm.channel_columns].sum() / total * 100
    controls = contrib[fleet.mmm.control_columns].sum() / total * 100 if fleet.mmm.control_columns else 0
    intercept = contrib['intercept'] / total * 100 if 'intercept' in contrib else 0
    
    results.append({'model': name, 'media_pct': media, 'controls_pct': controls, 'intercept_pct': intercept})

df1 = pd.DataFrame(results)

df
df1

###############################
## loo comparison NOT WORKING
################################

for name, fleet in [
    ("weibull_media_controls", pfleet0),
    ("geometric_controls", pfleet1), 
    ("geometric_no_controls", pfleet2),
    ("geometric_with_impressions", pfleet3),
    ("geometric_impressions_controls", pfleet4)
]:
    try:
        loo_result = az.loo(fleet.mmm.idata, pointwise=True)
        pareto_k = loo_result.pareto_k.values
        
        good = np.sum(pareto_k <= 0.7)
        bad = np.sum((pareto_k > 0.7) & (pareto_k <= 1.0))
        very_bad = np.sum(pareto_k > 1.0)
        
        print(f"\n{name}:")
        print(f"  Good (k≤0.7): {good}/{len(pareto_k)} ({good/len(pareto_k)*100:.1f}%)")
        print(f"  Bad (0.7<k≤1): {bad}/{len(pareto_k)} ({bad/len(pareto_k)*100:.1f}%)")
        print(f"  Very bad (k>1): {very_bad}/{len(pareto_k)} ({very_bad/len(pareto_k)*100:.1f}%)")
        print(f"  p_loo: {loo_result.p_loo:.1f}")
        print(f"  n_obs: {len(pareto_k)}")
    except Exception as e:
        print(f"{name}: Error - {e}")

print(f"\nNumPy version: {np.__version__}")
