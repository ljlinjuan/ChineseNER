from __future__ import annotations

import abc
from typing import TYPE_CHECKING
import pandas as pd
from datetime import datetime

if TYPE_CHECKING:
    from hqb.pipeline.single_factor.back_test_pipeline import SingleFactorBackTestPipelineABC


class SingleFactorBackTestResultAggregator(abc.ABC):
    def __init__(self) -> None:
        self.start_time = None
        self.finish_time = None
        self.neutralizer_parameter_coefficients_list = []
        self.factor_test_result_list = []
        self.correlation_df = pd.DataFrame()

    def start(self) -> SingleFactorBackTestResultAggregator:
        self.start_time = datetime.now()
        return self

    def add_factor_results(self, neutralizer_parameter_coefficients: pd.DataFrame, factor_test_result: SingleFactorBackTestPipelineABC.FactorTestResult):
        self.neutralizer_parameter_coefficients_list = neutralizer_parameter_coefficients
        self.factor_test_result_list.append(factor_test_result)

    def finish(self):
        self.finish_time = datetime.now()
        self.used_time = self.finish_time - self.start_time