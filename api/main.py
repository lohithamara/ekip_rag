from contextlib import asynccontextmanager
import traceback
import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Depends

from api.dependencies import create_container
from api.schemas import QueryRequest
from api.schemas import QueryResponse
from api.routes.auth import (
    router as auth_router,
)
from api.routes.documents import (
    router as documents_router,
)
from api.routes.departments import (
    router as departments_router,
)
from api.routes.users import (
    router as users_router,
)
from rag.service.schemas import RAGRequest
from security.authentication.schemas import AuthenticatedUser
from security.authentication.dependencies import get_current_user

from fastapi.middleware.cors import CORSMiddleware

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from sqlalchemy.exc import (
    OperationalError,
    SQLAlchemyError,
)

from qdrant_client.http.exceptions import (
    UnexpectedResponse,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    container = create_container()

    app.state.container = container

    try:
        yield

    finally:
        container.close()


app = FastAPI(
    title="EKIP RAG API",
    description=(
        "Multi-tenant enterprise "
        "knowledge intelligence API"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    auth_router
)

app.include_router(
    documents_router
)

app.include_router(
    departments_router
)

app.include_router(
    users_router
)

@app.get("/")
def root():

    return {
        "message": "EKIP RAG API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "EKIP RAG API",
        "version": "1.0.0",
    }

@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
    http_request: Request,

    current_user: AuthenticatedUser = Depends(
        get_current_user,
    ),
):

    try:

        rag_service = (
            http_request.app
            .state
            .container
            .rag_service
        )

        response = rag_service.answer(
            RAGRequest(

                query=request.query,

                user=current_user,

                conversation_id=request.conversation_id,
            )
        )

        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            metadata=response.metadata,
        )

    except UnexpectedResponse:

        logger.exception(
            "Qdrant request failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge retrieval service "
                "is temporarily unavailable. "
                "Please try again later."
            ),
        )

    except ConnectionError:

        logger.exception(
            "External service connection failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "A required service is temporarily "
                "unavailable. Please try again later."
            ),
        )

    except (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    ) as exc:

        logger.exception(
            "LLM service unavailable."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI generation service is "
                "temporarily unavailable. "
                "Please try again later."
            ),
        ) from exc

    except APIStatusError as exc:

        logger.exception(
            "LLM API request failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI generation service could "
                "not process the request. "
                "Please try again later."
            ),
        ) from exc

    except OperationalError as exc:

        logger.exception(
            "Database connection failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The database service is temporarily "
                "unavailable. Please try again later."
            ),
        ) from exc

    except SQLAlchemyError as exc:

        logger.exception(
            "Database operation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "A database error occurred while "
                "processing the request."
            ),
        ) from exc

    except Exception:

        logger.exception(
            "Unexpected error while processing query."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while "
                "processing your request."
            ),
        )