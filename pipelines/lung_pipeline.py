# pipelines/lung_pipeline.py
from pipelines.common_pipeline import OrganCTPipeline


class LungPipeline(OrganCTPipeline):
    def __init__(self) -> None:
        super().__init__("lung")
