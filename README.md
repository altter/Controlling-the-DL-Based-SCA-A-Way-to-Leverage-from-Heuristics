# Controlling the Deep Learning Based Side-Channel Aanlysis: A Way to Leverage from Heuristics
This repository contains a python base template* to perform a DoE gridsearch. It is aimed to find the optimal set of parameter for a deep learning model aimed to perform side-channel attack. It is based on the paper: 

**Controlling the Deep Learning-based Side-Channel Analysis: A Way to Leverage from Heuristics**

Servio Paguada, Unai Rioja, and Igor Armendariz

*_The user of this template is free to modified according his/her type of experiments._

### Implementation details

For implementation purposes, the parameters have been classified in three types: 
- Architecture parameters
- Training parameters
- Dataset parameters

`Architecure` parameters are those used to code/build the DL model (e.g. optimizer, layers and their parameters), `Training` parameters are those used in the training process (e.g. batch_size, epochs), and `Dataset` parameters represent sets or subsets of traces from the traces file.

### Example
_The example covered in this repository corresponds to the experiment of the iteration 4 of the paper._

The following code is a snap of the example in the `test.py` file.
To compose the experiments of the DoE, the class `DoEGridSearchSCA` recieves an instance of a class `Experiment` and a `ParameterTable`. The `Experiment` is an abstract class whose abstract method `do_doe()` is called from `DoEGridSearchSCA.do()` function. This latter receives the parameters requested by the DoE (three parameters).

The property `Results` in `DoEGridSearchSCA` contains the result of the experiments (as a python `dict` object).
The example runs one round of the experiment. Although, it is possible to specify the number of rounds using the parameter `rounds` in the `DoEGridSearchSCA` constructor.
In this template, fixed parameters are specified using two approaches (Both approaches are included in the example): 
1. As attributes of the Experiment class.
2. As attributes of the DoEDoEGridSearchSCA class. 

```python
###############################################################
# DoE parameter definitions
###############################################################
parameterA = Parameter('A', 'dim-of-fc-layer', 10, 20, parameter_type=ParameterType.ARCH_PARAM)
parameterB = Parameter('B', 'num-of-conv-layer', 1, 2, parameter_type=ParameterType.ARCH_PARAM)
parameterC = Parameter('C', 'num-of-fc-layer', 3, 4, parameter_type=ParameterType.ARCH_PARAM)
parameterD = Parameter('D', 'batch_size', 200, 400, parameter_type=ParameterType.TRAIN_PARAM)
parameterE = Parameter('E', 'epochs', 2, 5, parameter_type=ParameterType.TRAIN_PARAM)
parameterF = Parameter('F', 'original-original4FC', create_model(8), create_model(9), parameter_type=ParameterType.ARCH_PARAM)

###############################################################
# Parameter table
###############################################################
parameter_table = ParameterTable()
parameter_table.add_parameter(parameterA)
parameter_table.add_parameter(parameterB)
parameter_table.add_parameter(parameterC)
parameter_table.add_parameter(parameterD)
parameter_table.add_parameter(parameterE)
parameter_table.add_parameter(parameterF)

###############################################################
# Instantiating an experiment, metric, and a DoE
###############################################################
real_key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]

# Fixed parameter can be established as attibutes of the Experiment class,
# or using the attibuters of the DoEDoEGridSearchSCA class
expIteration4 = ExperimentIteration4(X_attack, real_key, plt_attack, 120, 75)
custom_metric = CustomMetric()

doe = DoEGridSearchSCA(metric=custom_metric, param_table=parameter_table, experiment=expIteration4)

###############################################################
# Setting the DoE
###############################################################
doe.ProfilingSet = [x_train_dict, y_train_dict, x_val_dict, y_val_dict]

###############################################################
# Run 'do' funtion to start the iterations
###############################################################
doe.do('A', 'B', 'C')
print (doe.Results)
print ('[INFO]: Saving file')
with open('doe_result.dat', 'wb') as handle:
    pickle.dump(doe.Results, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

`ExperimentIteration4` is a specification of the `Experiment` class. It requests, as parameters in its constructor, some fixed values to conduct the experiment.
```python
###############################################################
# The definition of the experiment, uses the Experiment base class
# It requires to override the do_doe function
###############################################################
class ExperimentIteration4(Experiment):
	def __init__(self, attack_set, real_key, plt_attack, batch_size, epochs):
		super().__init__()
		self.nb_traces_attacks = 300
		self.nb_attacks = 100
		self.real_key = real_key
		self.attack_set = attack_set
		self.plt_attack = plt_attack
		self.batch_size = batch_size
		self.epochs = epochs

	def do_doe(self, round, dataset_exp, arch_exp, train_exp, metric):
		print ('[INFO]: Round', round)
		train_exp = {'epochs':self.epochs, 'batch_size':self.batch_size}
		model = None
		model_str = str(arch_exp['dim-of-fc-layer']) + str(arch_exp['num-of-conv-layer']) + str(arch_exp['num-of-fc-layer'])
		print ()
		if model_str == '1013':
			model = create_model(0)
		if model_str == '1014':
			model = create_model(1)
		if model_str == '1023':
			model = create_model(2)
		if model_str == '1024':
			model = create_model(3)
		if model_str == '2013':
			model = create_model(4)
		if model_str == '2014':
			model = create_model(5)
		if model_str == '2023':
			model = create_model(6)
		if model_str == '2024':
			model = create_model(7)
		
		history = model.fit(dataset_exp[0], dataset_exp[1], validation_data=(dataset_exp[0], dataset_exp[1]), **train_exp)
		
		predictions = model.predict(self.attack_set)
		avg_rank = perform_attacks(self.nb_traces_attacks, predictions, self.nb_attacks, plt=self.plt_attack, key=self.real_key, byte=2, filename=model)
		return metric.compute(avg_rank)
```

A metric or score function has to be especified. The following is an implementation of the score function from the contribution of the mentioned paper

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
    return scoring_ge(GE, 256)
```
