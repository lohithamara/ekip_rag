from pathlib import Path
import shutil
import tempfile

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from fastapi.responses import FileResponse

from rag.documents.controller import (
    DocumentController,
)

from rag.documents.schemas import (
    DocumentInfo,
)

from rag.documents.upload_schema import (
    UploadDocumentResponse,
)

from rag.documents.exceptions import (
    DepartmentAccessDeniedError,
    DepartmentNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentError,
)

from security.authentication.dependencies import (
    get_current_user,
)

from security.authentication.schemas import (
    AuthenticatedUser,
)

router = APIRouter(
    prefix="/documents",
    tags=["Knowledge Base"],
)


def get_controller(
    request: Request,
) -> DocumentController:

    return DocumentController(
        request.app.state.container.document_service
    )


@router.post(
    "/upload",
    response_model=UploadDocumentResponse,
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department: str = Form(...),
    current_user: AuthenticatedUser = Depends(
        get_current_user,
    ),
):

    controller = get_controller(request)

    suffix = Path(file.filename).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:

        shutil.copyfileobj(
            file.file,
            temp,
        )

        temp_path = Path(temp.name)

    try:

        response = controller.upload_document(
            user=current_user,
            file_path=temp_path,
            original_filename=file.filename,
            department=department,
        )

        background_tasks.add_task(
            controller.service.worker.process_document,
            response.document_id,
        )

        return response

    except DuplicateDocumentError as exc:

        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )

    except DepartmentNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except DepartmentAccessDeniedError as exc:

        raise HTTPException(
            status_code=403,
            detail=str(exc),
        )

@router.get(
    "/{document_id}",
    response_model=DocumentInfo,
)
def get_document(
    document_id: int,
    request: Request,
    current_user: AuthenticatedUser = Depends(
        get_current_user,
    ),
):

    controller = get_controller(request)

    try:
        return controller.get_document(
            user=current_user,
            document_id=document_id,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except DepartmentAccessDeniedError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        )


@router.get(
    "",
    response_model=list[DocumentInfo],
)
def list_documents(
    request: Request,
    department: str | None = None,
    current_user: AuthenticatedUser = Depends(
        get_current_user,
    ),
):
    controller = get_controller(request)

    return controller.list_documents(
        user=current_user,
        department=department,
    )


@router.get(
    "/latest/{filename}",
    response_model=DocumentInfo,
)
def latest_version(
    filename: str,
    tenant_id: int,
    request: Request,
):

    controller = get_controller(request)

    try:
        return controller.latest_version(
            tenant_id=tenant_id,
            filename=filename,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

@router.get(
    "/download/{document_id}",
)
def download_document(
    document_id: int,
    request: Request,
    current_user: AuthenticatedUser = Depends(
        get_current_user,
    ),
):

    controller = get_controller(request)

    try:

        path = controller.download_document(
            user=current_user,
            document_id=document_id,
        )

        return FileResponse(
            path,
            filename=path.name,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except DepartmentAccessDeniedError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        )

@router.delete(
    "/{document_id}",
    status_code=204,
)
def delete_document(
    document_id: int,
    request: Request,
    current_user: AuthenticatedUser = Depends(
        get_current_user,
    ),
):
    controller = get_controller(request)

    try:
        controller.delete_document(
            user=current_user,
            document_id=document_id,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except DepartmentAccessDeniedError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        )