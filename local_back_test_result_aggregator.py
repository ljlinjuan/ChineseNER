from __future__ import annotations

import os
import shutil
import tempfile
import pandas as pd
from pathlib import Path
from hqb.pipeline.single_factor.back_test_pipeline import SingleFactorBackTestPipelineABC

from hqb.pipeline.single_factor.back_test_result_aggregator import SingleFactorBackTestResultAggregator
from hqb.visualizer.single_factor.aggregator.html_generator import AggregateBackTestResultHTMLGenerator
from hqb.visualizer.single_factor.detail.html_generator import SingleFactorBackTestResultHTMLGenerator


class LocalSingleFactorBackTestResultAggregator(SingleFactorBackTestResultAggregator):
    def __init__(self) -> None:
        super().__init__()
        self.local_cache_folder = os.path.join(tempfile.gettempdir(), "hqb_results", "single_factor", "local_aggregator")
        self.detail_html_relative_file_paths = []
        if os.path.exists(self.local_cache_folder):
            shutil.rmtree(self.local_cache_folder)
        Path(self.local_cache_folder).mkdir(parents=True)

    def add_factor_results(self, neutralizer_parameter_coefficients: pd.DataFrame, factor_test_result: SingleFactorBackTestPipelineABC.FactorTestResult):
        super().add_factor_results(neutralizer_parameter_coefficients, factor_test_result)
        detail_html_generator = SingleFactorBackTestResultHTMLGenerator(
            "{}: {}".format(factor_test_result.factor.__class__.__name__, factor_test_result.factor.name),
            neutralizer_parameter_coefficients,
            factor_test_result.coverage_result,
            factor_test_result.icir_result,
            factor_test_result.groups_result,
            factor_test_result.group_hedge_result,
            factor_test_result.index_hedge_result,
            local_cache_folder = os.path.join(self.local_cache_folder, "details"))
        self.detail_html_relative_file_paths.append(os.path.join("details", detail_html_generator.run()[1]))

    def finish(self):
        super().finish()
        aggregate_html_generator = AggregateBackTestResultHTMLGenerator(self)
        aggregate_html_generator.run()
