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

args = {'dataset': 'ijcnn1',
        'data_size': 49990,
        'n_features': 22,
        'reg_param': 1e-5,
        'seed': 0,
        'epochs': 10,
        'clean_data': False,
        'label_format': '-1 1'}

file_name = f"{ args['dataset'] }/{ args['dataset'] }"
try:
    load_data(file_name, args['data_size'], args['n_features'])
except:
    from urllib.request import urlretrieve
    import gzip
    import shutil
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2"
    urlretrieve(url, file_name+".bz2")
    import bz2
    zipfile = bz2.BZ2File(file_name+".bz2")
    data = zipfile.read()
    open(file_name, 'wb').write(data)

mice_params = {'eps': 0.5,
               'min_batch': 10,
               'max_hierarchy_size': 100,
               'restart_factor': 10,
               're_tot_cost': 1.,
               'aggr_cost': 2.0,
               're_min_n': 10,
               'adpt': False}

bay_params = {'penal': 1e-2,
              'reg_param': 1e-2}

bay_update_when = 5

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

make_plots('ijcnn1', 49990)
plot_newton_cg('ijcnn1')
plot_newton_cg('ijcnn1', 'svrg_bay')
plot_newton_cg('ijcnn1', 'sarah_bay')