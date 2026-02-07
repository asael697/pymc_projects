import pandas as pd

from meridian_toolkit.meridian_model import MeridianModel

media = ['meta_spend', 'google_spend', 'snapchat_spend', 
            'tiktok_spend', 'moloco_spend','liveintent_spend',
            'beehiiv_spend','amazon_spend']

controls = ["trend","seasonal"]

impressions = ['meta_impressions', 'google_impressions', 'snapchat_impressions', 
            'tiktok_impressions', 'moloco_impressions','liveintent_impressions',
            'beehiiv_impressions','amazon_impressions']


model1 = MeridianModel(
    client_data_path="data/monthly_mocha_ctrls.csv",
    date_var="date",
    target_name="subscriptions",
    depvar_type="non_revenue",
    channel_names=media,
    impression_names=impressions,
    control_names=None,
    adstock="binomial",
    lag_max=8,
    n_validate=4,
    time_varying_intercept=False,
    knots_percents=0.5)

model1.fit(n_draws=500,n_tune=500)
model1.mcmc_diagnostics

model1.mcmc_summary(var_names="roi_m")
model1.mcmc_plot(var_names="roi_m")

model1.get_channel_contribution()
model1.get_channel_roi()

model1.plot_predict(train_window=30,confidence_level=0.80)
model1.compute_validation_errors()



