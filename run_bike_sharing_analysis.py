from BikeSharingAnalysis import BikeSharingAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

bs_dataset_relative_path = "data/hour.csv"
bs_analyzer = BikeSharingAnalysis(bs_dataset_relative_path)

# choose a regressor, e.g. a DecisionTreeRegressor, SVR
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=12), n_estimators=300)
#regressor = SVR(C=500)
evaluation_result, model = bs_analyzer.run(regressor)
print evaluation_result
