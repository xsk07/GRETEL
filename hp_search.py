import sys

from src.dataset.dataset_factory import DatasetFactory
from src.oracle.oracle_factory import OracleFactory
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
    #context.factories['embedders'] = EmbedderFactory(context)
    #context.factories['explainers'] = ExplainerFactory(context)
    #context.factories['metrics'] = EvaluationMetricFactory(context.conf)
    #context.factories['plotters'] = PlotterFactory(context)

    hp_tuner = HpTuner(context, 3)
    hp_tuner.optimize()

