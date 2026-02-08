import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from meridian.analysis import visualizer
from statsmodels.tsa.seasonal import STL

from meridian_toolkit.meridian_model import MeridianModel

## load the data
data = pd.read_csv('data/monthly_mocha.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)

# visualize the depvar

plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['subscriptions'], linewidth=2, color='steelblue')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Subscriptions', fontsize=12)
plt.title('Subscriptions Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/depvar.png', dpi=300, bbox_inches='tight')
plt.close()

## Do a Time series decomposition to check the seasonality and trend

stl = STL(data['subscriptions'].values, seasonal=13, period = 4)
result = stl.fit()

result.plot()
plt.savefig('plots/stl_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()

## define control variables
control_data = pd.DataFrame({
    'trend': result.trend,
    'seasonal': result.seasonal,
})

# append them to the data
data = pd.concat([data,control_data], axis = 1)
data['revenue_per_kpi'] = data['subscriptions'] * 100
data.to_csv('data/monthly_mocha_ctrls.csv')

##########################################
# model definition
##########################################

media = ['meta_spend', 'google_spend', 'snapchat_spend', 
            'tiktok_spend', 'moloco_spend','liveintent_spend',
            'beehiiv_spend','amazon_spend']

# roku channel is removed because is a zero channel

controls = ["trend","seasonal"]

impressions = ['meta_impressions', 'google_impressions', 'snapchat_impressions', 
            'tiktok_impressions', 'moloco_impressions','liveintent_impressions',
            'beehiiv_impressions','amazon_impressions']


# Model definition list

#########################################################################
# Static models
#########################################################################

# This simple models do not contemplate time-varying intercept, neither 
# control variables meassuring the baselines dynamics, this both models
# are purely affected by the media market

# geometric
model1 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=None,
    adstock="geometric",
    lag_max=8,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.5)

model1.fit(n_draws=1000,n_tune=1000,cores=4)
model1.mcmc_diagnostics

model1.mcmc_summary(var_names="roi_m")
model1.mcmc_plot(var_names="roi_m")

model1.get_channel_contribution()
model1.get_channel_roi()

model1.plot_predict(train_window=30,confidence_level=0.80)
model1.compute_validation_errors()

# binomial
model11 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=None,
    adstock="binomial",
    lag_max=8,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.5)

model11.fit(n_draws=1000,n_tune=1000, seed = 98352)
model11.mcmc_diagnostics

model11.mcmc_summary(var_names="roi_m")
model11.mcmc_plot(var_names="roi_m")

model11.get_channel_contribution()
model11.get_channel_roi()

model11.plot_predict(train_window=30,confidence_level=0.80)
model11.compute_validation_errors()

# binomial vs geometric

results = {}
for name, model in [
    ("binomial_static", model1),
    ("geometric_static", model11),]:
    metrics = model.compute_validation_errors()
    results[name] = {
        'MAPE': metrics['mape'],
        'RMSE': metrics['rmse']
    }

df1 = pd.DataFrame(results).T
df1

# best model binomial

#########################################################################
# Time varying intercept
#########################################################################

# This models add time-varying intercept effect. by imposing a spline model
# over the interept. These models have two problems, Spline models overfit, 
# its hard to tune the models in terms of the number of knots, low knot values
# produces linear models as the static models ones, and high knot values produces
# complex non-linear models with low predictive power.

# time varying intercpet
model21 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=None,
    adstock="binomial",
    lag_max=8,
    n_validate=8,
    time_varying_intercept=True,
    knots_percents=0.7)

model21.fit(n_draws=1000,n_tune=1000)
model21.mcmc_diagnostics

model21.mcmc_summary(var_names="roi_m")
model21.mcmc_plot(var_names="roi_m")

model21.get_channel_contribution()
model21.get_channel_roi()

model21.plot_predict(train_window=30,confidence_level=0.80)
model21.compute_validation_errors()

# Geometric
model22 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=None,
    adstock="geometric",
    lag_max=8,
    n_validate=8,
    time_varying_intercept=True,
    knots_percents=0.7)

model22.fit(n_draws=1000,n_tune=1000, seed = 98352)
model22.mcmc_diagnostics

model22.mcmc_summary(var_names="roi_m")
model22.mcmc_plot(var_names="roi_m")

model22.get_channel_contribution()
model22.get_channel_roi()

model22.plot_predict(train_window=30,confidence_level=0.80)
model22.compute_validation_errors()

# binomial vs geometric
results = {}
for name, model in [
    ("binomial_TVI", model21),
    ("geometric_TVI", model22),]:
    metrics = model.compute_validation_errors()
    results[name] = {
        'MAPE': metrics['mape'],
        'RMSE': metrics['rmse']
    }

df2 = pd.DataFrame(results).T
df2

# best model: binomal adstock

#########################################################################
# Controls vars (Trend + seasonal)
#########################################################################

# This models mocks the time-varying intercept effect by adding the 
# trend and seasonal effects, values computed by a smoother model, in other
# words, a second simpler models that handles the process dynamics. 
#
# These kind of models produce baselines that canibalize the media contributions
# (baselines with a 80% or higher contributions) neglecting the media effect.

# binomial
model31 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=controls,
    adstock="binomial",
    lag_max=8,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.1)

model31.fit(n_draws=1000,n_tune=1000)
model31.mcmc_diagnostics

model31.mcmc_summary(var_names="roi_m")
model31.mcmc_plot(var_names="roi_m")

model31.get_channel_contribution()
model31.get_channel_roi()

model31.plot_predict(train_window=30,confidence_level=0.80)
model31.compute_validation_errors()

# Geometric
model32 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=controls,
    adstock="geometric",
    lag_max=8,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.1)

model32.fit(n_draws=1000,n_tune=1000)
model32.mcmc_diagnostics

model32.mcmc_summary(var_names="roi_m")
model32.mcmc_plot(var_names="roi_m")

model32.get_channel_contribution()
model32.get_channel_roi()

model32.plot_predict(train_window=30,confidence_level=0.80)
model32.compute_validation_errors()

# binomial vs geometric
results = {}
for name, model in [
    ("binomial_TVI", model31),
    ("geometric_TVI", model32),]:
    metrics = model.compute_validation_errors()
    results[name] = {
        'MAPE': metrics['mape'],
        'RMSE': metrics['rmse']
    }

df3 = pd.DataFrame(results).T
df3

# veredict: Binomial adstock

#################################
# Compare dynamic effects
##################################

# The following table compares all Binomial models
# according their different configurations, although
# the model with controls has the best model performance
# Static models produces competitive predictions with a more
# explainable contributions. 

## best model: Static Binomial adstock model

results = {}
for name, model in [
    ("Static", model11),
    ("TVI",   model21),
    ("Controls", model31),]:
    metrics = model.compute_validation_errors()
    divs = model.mcmc_diagnostics
    ctb = model.get_channel_contribution()["contributions"][0]
    results[name] = {
        'MAPE': metrics['mape'],
        'divergences': divs['n_divergences'],
        'baseline_ctb': float(ctb),
    }

df4 = pd.DataFrame(results).T
df4

####################################
# The effect of lag_max on the model
#####################################

model31_4 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=controls,
    adstock="binomial",
    lag_max=4,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.1)

model31_4.fit(n_draws=1000,n_tune=1000)


model31_10 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=controls,
    adstock="binomial",
    lag_max=10,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.1)

model31_10.fit(n_draws=1000,n_tune=1000)

model31_12 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=controls,
    adstock="binomial",
    lag_max=10,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.1)

model31_12.fit(n_draws=1000,n_tune=1000)

results = {}
for name, model in [
    ("lag_max_4", model31_4),
    ("lag_max_8", model31),
    ("lag_max_10",model31_10),
    ("lag_max_12",model31_12),]:
    ctb = model.get_channel_contribution()["contributions"][0]
    results[name] = {
        'baseline_ctb': float(ctb),
    }

df_lag_max = pd.DataFrame(results).T
df_lag_max

# The effect does not reduce when increasing the carryover effect. 

###########################################
# Best model: Static binomial full summary
############################################

model11 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=None,
    adstock="binomial",
    lag_max=8,
    n_validate=8,
    time_varying_intercept=False,
    knots_percents=0.5)

model11.fit(n_draws=1000,n_tune=1000, seed = 98352)
model11.mcmc_diagnostics

model11.mcmc_summary(var_names="roi_m")

## ROIs traceplots
ax = model11.mcmc_plot(var_names="roi_m")
fig = ax.ravel()[0].get_figure()
fig.savefig('plots/trace_plot_rois_static_binomial.png', dpi=300, bbox_inches='tight')
plt.close(fig)

## saturation curves
media_effects = visualizer.MediaEffects(model11.mmm)
chart = media_effects.plot_response_curves(confidence_level=0.50)
chart.save('plots/sat_curves_static_binomial.png', scale_factor=2.0)

## adstock effect
chart = media_effects.plot_adstock_decay(confidence_level=0.40,)
chart.save('plots/adstock_static_binomial.png', scale_factor=2.0)

## Contributions
model11.get_channel_contribution()

media_summary = visualizer.MediaSummary(model11.mmm)
chart = media_summary.plot_contribution_waterfall_chart()
chart.save('plots/contribution_static_binomial.png', scale_factor=2.0)

## ROAS
model11.get_channel_roi()
chart = media_summary.plot_roi_bar_chart()
chart.save('plots/roas_static_binomial.png', scale_factor=2.0)

### model's prediction
fig = model11.plot_predict(train_window=30,confidence_level=0.80)
fig.savefig('plots/predict_static_binomial.png', dpi=300, bbox_inches='tight')
plt.close(fig)

model11.compute_validation_errors()

## last months predictions
y_pred = model11.get_test_predictions()[:4]

# last months predicted subscriptions
y_pred['mean'] 

# predicted revenue
float(100 * y_pred['mean'].sum())

# last months actual revenue
float(100 * y_pred['observed'].sum())
