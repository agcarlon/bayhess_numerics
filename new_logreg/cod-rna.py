from logreg import load_data
from sgd import sgd
from sgd_mice import sgd_mice
from sgd_mice_bay import sgd_mice_bay
from svrg import svrg
from svrg_bay import svrg_bay
from sarah import sarah
from sarah_bay import sarah_bay
from make_plots import make_plots
from plot_newton_cg import plot_newton_cg

args = {'dataset': 'cod-rna',
        'data_size': 59535,
        'n_features': 8,
        'reg_param': 1e-5,
        'seed': 1,
        'epochs': 200,
        'clean_data': False,
        'label_format': '-1 1'}

file_name = f"{ args['dataset'] }/{ args['dataset'] }"
try:
    load_data(file_name, args['data_size'], args['n_features'])
except:
    from urllib.request import urlretrieve
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna"
    urlretrieve(url, file_name)


mice_params = {'eps': 0.5,
               'min_batch': 5,
               'max_hierarchy_size': 200,
               'restart_factor': 10,
               're_tot_cost': 1.,
               're_min_n': 10,
               'adpt': False}

bay_params = {'penal': 1e-2,
              'reg_param': 1e-2,
              'pairs_to_use': 80,
              'verbose': False}

bay_update_when = 40

svrg_step = 0.1
svrg_batch_size = 5

# sgd(**args)
# sgd_mice(**args, mice_params=mice_params)
sgd_mice_bay(**args, update_when=bay_update_when, mice_params=mice_params,
             bay_params=bay_params)
plot_newton_cg('cod-rna', 'sgd_mice_bay')

# svrg(**args, step_param=svrg_step, batch_size=svrg_batch_size)
svrg_bay(**args, step_param=svrg_step, batch_size=svrg_batch_size,
         update_when=bay_update_when, bay_params=bay_params)
plot_newton_cg('cod-rna', 'svrg_bay')

# sarah(**args, step_param=svrg_step, batch_size=svrg_batch_size)
sarah_bay(**args, step_param=svrg_step, batch_size=svrg_batch_size,
          update_when=bay_update_when, bay_params=bay_params)
plot_newton_cg('cod-rna', 'sarah_bay')


make_plots('cod-rna', 59535)
