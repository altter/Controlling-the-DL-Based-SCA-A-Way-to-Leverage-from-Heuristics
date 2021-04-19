# Controlling the Deep Learning Based Side-Channel Aanlysis: A Way to Leverage from Heuristics
This repository contains a python base template to perform a DoE gridsearch to find the optimal set of parameter for a deep learning model aimed to perform side-channel attack. It is based on the paper: 

**Controlling the Deep Learning-based Side-Channel Analysis: A Way to Leverage from Heuristics**

Servio Paguada, Unai Rioja, and Igor Armendariz

*The user of this template is free to modified according his/her type of experiments.

### Implementation details

For implementation purposes, the parameters have been classified in three types: 
- Architecture parameters
- Training parameters
- Dataset parameters

`Architecure` parameters are those used to code/build the DL model (e.g. optimizer, layers and their parameters), `Training` parameters are those used in the trainig process (e.g. batch_size, epochs), and `Dataset` parameters represent sets or subset of traces from the traces file.

To compose the experiments of the DoE, the class `DoEGridSearchSCA` recieves a instance of a class `Experiments` and a `ParameterTable`. The Experiment is an abstract class whose abstract method `do_doe` is called from `DoEGridSearchSCA.do()` function. This latter receives the parameters requested by the DoE (three parameters).
The following code is a snap of the example in the `test.py` file. 
The property `Result` in `DoEGridSearchSCA` contains the result of the experiments.
The example runs three rounds of each experiments to compute the average.
```python
parameter = Parameter('A', 'batch_size', 200, 400, parameter_type=ParameterType.TRAIN_PARAM)
parameter2 = Parameter('B', 'epochs', 50, 100, parameter_type=ParameterType.TRAIN_PARAM)
parameter3 = Parameter('C', 'optimizer', create_model(1), create_model(2), parameter_type=ParameterType.ARCH_PARAM)

###############################################################
# Parameter table creation
###############################################################
parameter_table = ParameterTable()
parameter_table.add_parameter(parameter)
parameter_table.add_parameter(parameter2)
parameter_table.add_parameter(parameter3)

###############################################################
# Instantiating an experiment, metric, and a DoE
###############################################################
real_key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]

my_exp = MyExperiment(X_attack, real_key, plt_attack)
custom_metric = CustomMetric()

doe = DoEGridSearchSCA(metric=custom_metric, param_table=parameter_table, experiment=my_exp, rounds=3)

###############################################################
# Setting the DoE
###############################################################
doe.ProfilingSet = [x_train_dict, y_train_dict, x_val_dict, y_val_dict]

###############################################################
# Run 'do' funtion to start the iterations
###############################################################
doe.do('A', 'B', 'C')
print (doe.Results)
```

`MyExperiment` is a specification of the Experiment class. It requests, as parameters in its constructor, some fixed values to conduct the experiment.
```python
###############################################################
# The definition of the experiment, uses the Experiment base class
# It requires to override the do_doe function
###############################################################
class MyExperiment(Experiment):
  def __init__(self, attack_set, real_key, plt_attack):
    super().__init__()
    self.i = 0
    self.nb_traces_attacks = 300
    self.nb_attacks = 100
    self.real_key = real_key
    self.attack_set = attack_set
    self.plt_attack = plt_attack

  def do_doe(self, dataset_exp, arch_exp, train_exp, metric):
    print ('[INFO]: iteration', self.i + 1)
    model = arch_exp['optimizer']
    history = model.fit(dataset_exp[0], dataset_exp[1], validation_data=(dataset_exp[0], dataset_exp[1]), **train_exp)

    predictions = model.predict(self.attack_set)
    avg_rank = perform_attacks(self.nb_traces_attacks, predictions, self.nb_attacks, plt=self.plt_attack, key=self.real_key, byte=2, filename=model)
    self.i += 1
    return metric.compute(avg_rank)
```

A metric or score function has to be especified. The following is an implementation of the score function in the contribution of the paper

```python
###############################################################
# Metric definition - implemented using a class (it could be a function)
###############################################################

def trend_diffsum(GE, i=0, value=0):
  if len(GE) == (i+1):
    return value
  return trend_diffsum(GE, i+1, value + ((GE[i])-(GE[i+1])))

def compute_trend(GE):
  value = trend_diffsum(GE)
  if value == 0:
    GE.insert(0, 256)
    value = trend_diffsum(GE)
  trend = ((value - min(GE)) / (max(GE) - min(GE)))
  return trend

def scoring_ge(GE, M):
  trend = trend_diffsum(GE)
  return (trend/2) * (((M - GE[-1])/M) + ((len(GE) - np.argmin(GE)+1)/len(GE)))

class CustomMetric():
  def __init__(self) -> None:
    super().__init__()
    self.__value = 0

  def compute(self, GE):
    print ('[INFO]: Computing score function')
    return scoring_ge(GE)
```
