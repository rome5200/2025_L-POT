# pipelines/liver_pipeline.py
from pipelines.common_pipeline import OrganCTPipeline


class LiverPipeline(OrganCTPipeline):
    def __init__(self) -> None:
        super().__init__("liver")
