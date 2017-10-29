This folder contains preliminary test results for implementing the momentum factor from winning or losing streaks into the team's 
current game prediction models. As a starter, I tested how to calculate and choose hyperparameters that represent the momentum factor 
in a real life sense. The test code is momentum.py which incorporates function to calculate momentum is a previous period from the current
game date. Boston Celtics was chosen as a test case and the momentum time series for its 2016-2017 season results was plotted using 
different parameters. In particular, rolling winsow sizes k=10 and k=5 are tested and discount factors d=0.9 and d=0.5 are used.
See the plots for results.
