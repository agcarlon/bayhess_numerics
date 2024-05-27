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


args = {'dataset': 'mushrooms',
        'data_size': 8124,
        'n_features': 112,
        'reg_param': 1e-5,
        'seed': 0,
        'epochs': 100,
        'clean_data': False,
        'label_format': '1 2'}

file_name = f"{ args['dataset'] }/{ args['dataset'] }"
try:
    load_data(file_name, args['data_size'], args['n_features'])
except:
    from urllib.request import urlretrieve
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
    urlretrieve(url, file_name)

mice_params = {'eps': 0.5,
               'min_batch': 5,
               'max_hierarchy_size': 100,
               'restart_factor': 10,
               're_tot_cost': 1.,
               're_min_n': 10,
               'aggr_cost': 0.5,
               # 'mice_type': 'light',
               'adpt': False}

bay_params = {'penal': 1e-2,
              'reg_param': 1e-2}


bay_update_when = 25

svrg_step = 0.1
svrg_batch_size = 5


sgd(**args)
sgd_mice(**args, mice_params=mice_params)
sgd_mice_bay(**args, update_when=bay_update_when, mice_params=mice_params,
             bay_params=bay_params)
svrg(**args, step_param=svrg_step, batch_size=svrg_batch_size)
svrg_bay(**args, step_param=svrg_step, batch_size=svrg_batch_size,
         update_when=bay_update_when, bay_params=bay_params)
sarah(**args, step_param=svrg_step, batch_size=svrg_batch_size)
sarah_bay(**args, step_param=svrg_step, batch_size=svrg_batch_size,
          update_when=bay_update_when, bay_params=bay_params)

make_plots('mushrooms', 8124)
plot_newton_cg('mushrooms')
plot_newton_cg('mushrooms', 'svrg_bay')
plot_newton_cg('mushrooms', 'sarah_bay')
