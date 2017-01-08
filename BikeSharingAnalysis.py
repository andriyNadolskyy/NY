import os
import subprocess
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class BikeSharingAnalysis:

    def __init__(self, relative_dataset_path):
        self.relative_dataset_path = relative_dataset_path
        self.df = pd.DataFrame()

    def _visualize_tree(self, tree, data_columns, feature_importances):
        """Create tree png using graphviz

        Args:
            tree: scikit-learn DecisionTree
            data_columns: data columns
            feature_importances: importance of features coming from DecisionTreeRegressor
        """
        for i in range(11):
            print str(data_columns[i]) + " " + str(feature_importances[i])

        with open("dt.dot", 'w') as f:
            feature_names = list(data_columns[0:11])
            export_graphviz(tree, out_file=f, feature_names=feature_names)

        command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
        try:
            subprocess.check_call(command)
        except:
            print("Could not export a decision tree in .dot format. Check your graphviz installation.")

    def _read_data(self):
        """Read a .csv file as a data frame
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(current_path, self.relative_dataset_path)
        self.df = self.df.from_csv(dataset_path)

    def _prepare_dataset(self):
        """Remove columns from the data frame:
            'casual', 'registered': not allowed to be used
            'dteday', 'yr': currently useless for future predicting
        """
        self.df.drop('casual', axis=1, inplace=True)
        self.df.drop('registered', axis=1, inplace=True)
        self.df.drop('dteday', axis=1, inplace=True)
        self.df.drop('yr', axis=1, inplace=True)

    def _train_model(self, data, regressor):
        """Train a model for given data using the specified regressor

        Args:
            data: data containing features and gold label
            regressor: given regressor

        Returns:
            trained model
        """
        data_columns = list(data.columns)
        training_data = data[data_columns[0:11]]
        gold_labels = list(data["cnt"])

        # Fit regression model
        model = regressor.fit(training_data, gold_labels)

        # produce results for table 1 and figure 4 of documentation
#        tree_feature_importances = list(model.feature_importances_)
#        self._visualize_tree(model, data_columns, tree_feature_importances)
        return model

    def _evaluate_model(self, model, data):
        """Evaluate a trained model by mean absolute error

        Args:
            model: trained model
            data: data containing features and gold label

        Returns:
            value of mean absolute error
        """
        data_columns = list(data.columns)
        test_data = data[data_columns[0:11]]
        gold_labels = list(data["cnt"])
        predictions = model.predict(test_data)
        return mean_absolute_error(gold_labels, predictions)

    def run(self, regressor):
        """Run complete data analysis pipeline:
        read data, clean data, divide dataset into train and test dataset,
        train a model, evaluate the trained model

        Args:
            regressor: given regressor

        Returns:
            evaluation result for the trained model
        """
        self._read_data()
        self._prepare_dataset()

        train, test = train_test_split(self.df, test_size=0.1, random_state=23)
        trained_model = self._train_model(train, regressor)

        evaluation_result = self._evaluate_model(trained_model, test)
        return evaluation_result, trained_model
