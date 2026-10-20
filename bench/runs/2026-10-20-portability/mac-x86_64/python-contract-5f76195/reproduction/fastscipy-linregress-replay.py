import importlib.util
import json
import warnings
import numpy as np
import scipy.stats as st
spec = importlib.util.spec_from_file_location('stats_tests', 'tests/test_stats.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for threads in (1, 4):
    for name, generate in m.CASES.items():
        for iteration in range(150):
            args, kwargs = generate()
            if (threads, name, iteration) != (4, 'linregress', 69):
                continue
            x, y = args
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = st.linregress(x, y, **kwargs)
            report = dict(seed=12345, threads=threads, function=name, iteration=iteration,
                          kwargs=kwargs, x=x.tolist(), y=y.tolist(), x_dtype=str(x.dtype), y_dtype=str(y.dtype),
                          xmean=float(x.mean()), xspread=float(np.max(np.abs(x-x.mean()))),
                          result={k:float(getattr(result,k)).hex() for k in (*result._fields,'intercept_stderr')},
                          warnings=[(v.category.__name__,str(v.message)) for v in caught])
            print(json.dumps(report, indent=2))
