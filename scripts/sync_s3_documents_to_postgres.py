from pathlib import Path
import hashlib

from sqlalchemy import select

from database.session import SessionLocal

from database.models.tenant import Tenant
from database.models.department import Department
from database.models.document import Document

from ingestion.storage.s3_client import S3Client


TENANT_NAME = "EKIP"

VALID_DEPARTMENTS = {
    "finance",
    "hr",
    "engineering",
    "legal",
    "sales",
    "support",
}


def calculate_s3_hash(
    s3_client: S3Client,
    s3_key: str,
) -> str:
    """
    Calculate SHA-256 directly from the S3
    object body without saving it permanently.
    """

    response = s3_client.client.get_object(
        Bucket=s3_client.bucket_name,
        Key=s3_key,
    )

    hasher = hashlib.sha256()

    body = response["Body"]

    while True:

        chunk = body.read(
            1024 * 1024
        )

        if not chunk:
            break

        hasher.update(
            chunk
        )

    return hasher.hexdigest()


def main():

    session = SessionLocal()

    s3_client = S3Client()

    try:

        # -----------------------------------------
        # 1. FIND TENANT
        # -----------------------------------------

        tenant = session.scalar(

            select(Tenant)

            .where(
                Tenant.name
                == TENANT_NAME
            )
        )

        if tenant is None:

            raise ValueError(
                f"Tenant '{TENANT_NAME}' "
                "not found."
            )


        # -----------------------------------------
        # 2. LOAD DEPARTMENTS
        # -----------------------------------------

        departments = {

            department.name:
                department

            for department

            in session.scalars(

                select(Department)

                .where(
                    Department.tenant_id
                    == tenant.id
                )

            ).all()

        }


        # -----------------------------------------
        # 3. LIST S3 OBJECTS
        # -----------------------------------------

        objects = (
            s3_client.list_objects(
                prefix=""
            )
        )


        print()
        print("=" * 70)
        print("S3 -> POSTGRES DOCUMENT SYNC")
        print("=" * 70)

        print(
            f"S3 objects discovered: "
            f"{len(objects)}"
        )


        created = 0
        skipped = 0
        ignored = 0
        failed = 0


        # -----------------------------------------
        # 4. PROCESS OBJECTS
        # -----------------------------------------

        for obj in objects:

            s3_key = obj["Key"]

            # Ignore folder-like objects

            if s3_key.endswith("/"):
                continue


            try:

                parts = Path(
                    s3_key
                ).parts


                # ---------------------------------
                # FIND DEPARTMENT FROM S3 KEY
                # ---------------------------------

                department_name = None

                for part in parts:

                    normalized = (
                        part
                        .lower()
                        .strip()
                    )

                    if (
                        normalized
                        in VALID_DEPARTMENTS
                    ):

                        department_name = (
                            normalized
                        )

                        break


                if department_name is None:

                    print(
                        f"IGNORED: {s3_key} "
                        "(department not found)"
                    )

                    ignored += 1

                    continue


                department = (
                    departments.get(
                        department_name
                    )
                )


                if department is None:

                    print(
                        f"IGNORED: {s3_key} "
                        "(department missing "
                        "in database)"
                    )

                    ignored += 1

                    continue


                filename = Path(
                    s3_key
                ).name


                # ---------------------------------
                # CHECK S3 KEY ALREADY EXISTS
                # ---------------------------------

                existing = session.scalar(

                    select(Document)

                    .where(
                        Document.s3_key
                        == s3_key
                    )
                )


                if existing:

                    print(
                        f"SKIPPED: {filename}"
                    )

                    skipped += 1

                    continue


                # ---------------------------------
                # CONTENT HASH
                # ---------------------------------

                print(
                    f"Processing: {filename}"
                )

                content_hash = (
                    calculate_s3_hash(
                        s3_client,
                        s3_key,
                    )
                )


                # ---------------------------------
                # CREATE POSTGRES RECORD
                # ---------------------------------

                document = Document(

                    filename=filename,

                    file_type=(
                        Path(filename)
                        .suffix
                        .lower()
                    ),

                    content_hash=(
                        content_hash
                    ),

                    file_size=(
                        obj.get(
                            "Size",
                            0,
                        )
                    ),

                    s3_key=s3_key,

                    tenant_id=tenant.id,

                    department_id=(
                        department.id
                    ),

                    # Historical document:
                    # no UI uploader available.

                    uploaded_by=None,

                    version=1,

                    # Already ingested previously.

                    status="COMPLETED",
                )


                session.add(
                    document
                )

                session.commit()


                print(
                    f"CREATED: "
                    f"{department_name}/"
                    f"{filename}"
                )

                created += 1


            except Exception as exc:

                session.rollback()

                failed += 1

                print(
                    f"FAILED: {s3_key}"
                )

                print(
                    f"Reason: {exc}"
                )


        # -----------------------------------------
        # 5. SUMMARY
        # -----------------------------------------

        print()
        print("=" * 70)
        print("SYNC COMPLETE")
        print("=" * 70)

        print(
            f"Created : {created}"
        )

        print(
            f"Skipped : {skipped}"
        )

        print(
            f"Ignored : {ignored}"
        )

        print(
            f"Failed  : {failed}"
        )

        print("=" * 70)


    finally:

        session.close()


if __name__ == "__main__":

    main()