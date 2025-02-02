import logging
import transformers

from fastapi import FastAPI
from contextlib import asynccontextmanager
from models import ScoreRequest, Score, RougeScore

from rouge_score import rouge_scorer
from bert_score import score

scorer = None

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    global scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    yield  # Finish initialization


app = FastAPI(lifespan=lifespan)


@app.post("/rogue_score")
async def rogue_score(request: ScoreRequest) -> RougeScore:
    global scorer
    scores = scorer.score(request.reference, request.candidate)
    return RougeScore(
        rouge1=Score(
            precision=scores["rouge1"][0],
            recall=scores["rouge1"][1],
            f1=scores["rouge1"][2],
        ),
        rouge2=Score(
            precision=scores["rouge2"][0],
            recall=scores["rouge2"][1],
            f1=scores["rouge2"][2],
        ),
        rougeL=Score(
            precision=scores["rougeL"][0],
            recall=scores["rougeL"][1],
            f1=scores["rougeL"][2],
        ),
    )


@app.post("/bert_score")
async def bert_score(request: ScoreRequest) -> Score:
    p, r, f1 = score(
        cands=[request.candidate],
        refs=[request.reference],
        lang="en",
        rescale_with_baseline=True,
    )
    return Score(precision=p[0], recall=r[0], f1=f1[0])
