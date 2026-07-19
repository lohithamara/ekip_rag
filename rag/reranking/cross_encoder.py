import time

from sentence_transformers import CrossEncoder

from rag.reranking.base import BaseReranker
from rag.reranking.config import RerankerConfig
from rag.reranking.schemas import (
    RerankRequest,
    RerankResponse,
    RerankedResult,
)


class CrossEncoderReranker(BaseReranker):

    def __init__(
        self,
        config: RerankerConfig,
    ):

        self.config = config

        self.model = CrossEncoder(
            model_name=config.model_name,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
        )

    def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:

        start_time = time.perf_counter()

        if not request.results:

            return RerankResponse(
                results=[],
                model_name=self.config.model_name,
                total_candidates=0,
                returned_candidates=0,
                elapsed_seconds=0.0,
            )

        pairs = [
            (
                request.query,
                result.text,
            )
            for result in request.results
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
        )

        ranked = sorted(
            zip(
                request.results,
                scores,
                strict=True,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        top_score = float(
            ranked[0][1]
        )

        relative_threshold = (
            top_score * 0.10
        )

        filtered_ranked = [
            (result, score)
            for result, score in ranked
            if float(score)
            >= relative_threshold
        ]

        # print("\n" + "=" * 80)
        # print("RERANKING RESULTS")
        # print("=" * 80)

        # for rank, (result, score) in enumerate(
        #     ranked,
        #     start=1,
        # ):
        #     print(
        #         f"{rank}. "
        #         f"Score: {float(score):.4f} | "
        #         f"Source: "
        #         f"{result.metadata.get('source_filename')}"
        #     )

        # print("=" * 80)
        reranked_results = [

            RerankedResult(
                retrieval_result=result,
                rerank_score=float(score),
            )

            for result, score in filtered_ranked[
                : request.top_k
            ]
        ]

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return RerankResponse(

            results=reranked_results,

            model_name=self.config.model_name,

            total_candidates=len(
                request.results
            ),

            returned_candidates=len(
                reranked_results
            ),

            elapsed_seconds=elapsed,
        )