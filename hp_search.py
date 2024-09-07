import sys

from src.dataset.dataset_factory import DatasetFactory
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.explainer.explainer_factory import ExplainerFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory
from src.plotters.plotter_factory import PlotterFactory
from src.utils.context import Context
from hpsearch.hp_tuner import HpTuner


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hp_search.py <config_file> [run_number]")
        sys.exit(1)

    context = Context.get_context(sys.argv[1])
    context.run_number = int(sys.argv[2]) if len(sys.argv) == 3 else -1

    context.factories['datasets'] = DatasetFactory(context)
    context.factories['oracles'] = OracleFactory(context)
    context.factories['embedders'] = EmbedderFactory(context)
    context.factories['explainers'] = ExplainerFactory(context)
    context.factories['metrics'] = EvaluationMetricFactory(context.conf)
    context.factories['plotters'] = PlotterFactory(context)

    hp_tuner = HpTuner(context, train_oracle=True, train_expainer=False, n_trials=100)
    hp_tuner.optimize()

