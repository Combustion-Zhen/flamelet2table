def params2name(params):
    params_str = []
    for k, v in params.items():
        try:
            param_str = '{0}-{1:g}'.format(k,v)
        except ValueError:
            param_str = '{0}-{1}'.format(k,v)
        params_str.append(param_str)
    return '_'.join(params_str)

def name2params(name):
    params = {}
    params_str = name.split('_')
    for param_str in params_str:
        param = param_str.split('-',maxsplit=1)
        key = param[0]
        try:
            params[key] = float( param[1] )
        except ValueError:
            params[key] = param[1]
    return params
